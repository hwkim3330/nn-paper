#!/usr/bin/env python3
"""
신경망 자동매매 시스템 — 메인 진입점
- 후회 기반 학습: 매매 결과를 분석해서 다음에 더 잘하기
- 모의투자 무한 반복 → 기준 충족 시 실전 전환

사용법:
  python3 nn_main.py --backfill       # 과거 데이터 백필 (최초 1회)
  python3 nn_main.py --features       # 피처 계산
  python3 nn_main.py --train          # 모델 학습
  python3 nn_main.py --paper          # 모의투자 시작
  python3 nn_main.py --status         # 현재 상태 조회
  python3 nn_main.py --report         # 텔레그램 리포트 즉시 발송
  python3 nn_main.py --dashboard      # 웹 대시보드 시작
"""
import sys
import os
import time
import logging
import argparse
from datetime import datetime

# stock-bot 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nn_config import (
    DB_PATH, MODEL_DIR, NN_LOG_FILE, LOG_FORMAT,
    ALL_STOCKS, SECTOR_MAP, PAPER_INITIAL_CAPITAL,
    SEQUENCE_LENGTH, BACKFILL_DAYS,
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(NN_LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nn_main")


def get_api():
    """KiwoomAPI 인스턴스."""
    from kiwoom_api import KiwoomAPI
    return KiwoomAPI()


def get_db():
    """Database 인스턴스."""
    from db import Database
    return Database(str(DB_PATH))


def _parse_market_stocks(result):
    """API 결과에서 종목 파싱 (ETF/우선주/외국주/동전주 제외).

    모멘텀 스코어 = 등락률(%) × log10(거래대금/억)
    단, +15% 이상 급등주는 패널티 (고점 추격 방지)
    """
    import math
    rows = result.get('trde_qty_updt', [])
    parsed = []
    for r in rows:
        code = str(r.get('stk_cd', '')).strip()
        name = str(r.get('stk_nm', '')).strip()
        if not code or len(code) != 6 or not code.isdigit():
            continue
        if any(name.startswith(p) for p in (
            'TIGER', 'KODEX', 'KBSTAR', 'RISE', 'PLUS', 'ACE', 'SOL',
            'HANARO', '마이티', 'ARIRANG', 'BNK', 'FOCUS', 'TIMEFOLIO',
        )):
            continue
        if code.startswith('9'):
            continue
        # 우선주 필터 (끝자리 5,7,8,9 AND 끝에서 두번째 0)
        if code[-1] in ('5', '7', '8', '9') and code[-2] == '0':
            continue

        prc_s = str(r.get('cur_prc', '0')).replace('+', '').replace('-', '').replace(',', '').strip() or '0'
        vol_s = str(r.get('now_trde_qty', '0')).replace(',', '').strip() or '0'
        rate_s = str(r.get('flu_rt', '0')).replace('+', '').strip() or '0'
        try:
            price = int(prc_s)
            volume = int(vol_s)
            rate = float(rate_s)
        except ValueError:
            continue
        if price < 2000 or volume < 10000:
            continue

        amount = price * volume
        # 모멘텀 스코어
        if amount > 0 and rate > 0.5:
            momentum = rate * math.log10(max(amount / 1e8, 1))

            # 고점 추격 패널티: +15% 이상 이미 급등한 종목은 모멘텀 절반
            if rate > 15.0:
                momentum *= 0.3
            elif rate > 10.0:
                momentum *= 0.6

            # 대형주 보너스: 거래대금 500억+ → 시장 주도주
            if amount > 50e9:
                momentum *= 1.5
            elif amount > 20e9:
                momentum *= 1.2
        else:
            momentum = 0

        parsed.append({
            'code': code, 'name': name, 'price': price,
            'volume': volume, 'rate': rate, 'amount': amount,
            'momentum': momentum,
        })
    return parsed


# 모멘텀 스캔 결과를 전역으로 공유 (predict_and_trade에서 활용)
_latest_momentum = {}
_prev_momentum = {}  # 이전 스캔 대비 모멘텀 변화 추적
_us_sentiment = 0.0  # 미국장 센티먼트 (-1.0 ~ +1.0)


def detect_volume_surge(db, code: str, current_volume: int) -> float:
    """거래량 폭증 감지. 20일 평균 대비 비율 리턴.

    3배 이상 → 뭔가 있다 (기관/외인 진입, 뉴스, 테마)
    5배 이상 → 확실한 시그널
    """
    rows = db.execute_raw(
        "SELECT volume FROM daily_prices WHERE stock_code=? "
        "ORDER BY date DESC LIMIT 20", (code,)
    )
    if not rows or len(rows) < 5:
        return 1.0
    avg_vol = sum(r['volume'] for r in rows if r['volume']) / len(rows)
    if avg_vol <= 0:
        return 1.0
    return current_volume / avg_vol


def detect_orderbook_imbalance(db, code: str) -> float:
    """호가 불균형 감지. 매수잔량/매도잔량 비율.

    > 1.5 → 매수세 강함 (올라갈 확률 높음)
    < 0.7 → 매도세 강함 (내려갈 확률 높음)
    """
    rows = db.execute_raw(
        "SELECT ask_bid_ratio FROM orderbook_snapshots "
        "WHERE stock_code=? ORDER BY id DESC LIMIT 1", (code,)
    )
    if not rows:
        return 1.0
    return float(rows[0].get('ask_bid_ratio', 1.0) or 1.0)


def check_us_sentiment(db) -> float:
    """미국장 센티먼트 계산 (-1.0 ~ +1.0).

    S&P500/NASDAQ 전일 종가 기준:
    - 양수: 미국장 상승 → 한국장에 호재
    - 음수: 미국장 하락 → 한국장에 악재
    """
    global _us_sentiment
    try:
        rows = db.execute_raw(
            "SELECT symbol, change_pct FROM market_data "
            "WHERE date = (SELECT MAX(date) FROM market_data) "
            "AND symbol IN ('SP500', 'NASDAQ', 'VIX')"
        )
        if not rows:
            return 0.0

        sentiment = 0.0
        for r in rows:
            sym = r['symbol']
            pct = float(r.get('change_pct', 0) or 0)
            if sym in ('SP500', 'NASDAQ'):
                sentiment += pct * 0.3  # 양수면 호재
            elif sym == 'VIX':
                sentiment -= pct * 0.1  # VIX 상승은 악재

        _us_sentiment = max(-1.0, min(1.0, sentiment / 3))
        logger.info("US 센티먼트: %.2f (SP500/NQ/VIX 기반)", _us_sentiment)
        return _us_sentiment
    except Exception as e:
        logger.error("US 센티먼트 계산 실패: %s", e)
        return 0.0


def get_dynamic_watchlist(api, db, portfolio=None):
    """거래대금 상위 전 종목 스캔 (200~300종목).

    전략:
    1. 코스피+코스닥 거래대금 TOP 전부 수집 (API 2번)
    2. 가격/등락률/거래량 bulk 데이터 활용 (개별 API콜 불필요)
    3. 모멘텀 양수 종목 전부 신경망 투입
    4. 보유 종목 반드시 포함
    5. 신규 종목 백필은 주기당 최대 10개 제한
    """
    global _latest_momentum
    logger.info("전 종목 스캔 (코스피+코스닥 거래대금 TOP)...")

    # 코스피 거래대금 순위 (보통 100~150종목)
    result = api.stock_info.trading_volume_update_request_ka10024(
        market_type='0', cycle_type='1',
        trade_quantity_type='0', stock_exchange_type='0',
    )
    parsed_kospi = _parse_market_stocks(result) if result else []

    # 코스닥 거래대금 순위 (보통 80~120종목)
    try:
        result2 = api.stock_info.trading_volume_update_request_ka10024(
            market_type='1', cycle_type='1',
            trade_quantity_type='0', stock_exchange_type='0',
        )
        parsed_kosdaq = _parse_market_stocks(result2) if result2 else []
    except Exception:
        parsed_kosdaq = []

    # 중복 제거 (코드 기준)
    seen = set()
    parsed = []
    for s in parsed_kospi + parsed_kosdaq:
        if s['code'] not in seen:
            seen.add(s['code'])
            parsed.append(s)

    if not parsed:
        logger.warning("거래량 API 실패 — 기존 워치리스트 사용")
        from config import WATCHLIST
        return list(WATCHLIST.keys())

    # 전부 저장 (가격 데이터 bulk으로 활용)
    _latest_momentum = {s['code']: s for s in parsed}

    # 모멘텀 양수 전부 + 거래대금 상위 = 전부 신경망 투입
    parsed.sort(key=lambda x: x['momentum'], reverse=True)
    top_codes = [s['code'] for s in parsed if s['momentum'] > 0]

    # 모멘텀 0이어도 거래대금 상위면 포함 (감시 대상)
    for s in parsed:
        if s['code'] not in top_codes and s['amount'] > 5e9:
            top_codes.append(s['code'])

    # 보유 종목 반드시 포함
    if portfolio and portfolio.positions:
        for code in list(portfolio.positions.keys()):
            if code not in top_codes:
                top_codes.append(code)

    # 신규 종목 일봉 백필 (주기당 최대 10개)
    from collector.price_collector import PriceCollector
    price_collector = PriceCollector(api, db)
    backfill_count = 0
    for code in top_codes:
        if backfill_count >= 10:
            break
        existing = db.execute_raw(
            "SELECT COUNT(*) as cnt FROM daily_prices WHERE stock_code=?", (code,)
        )
        if not existing or existing[0]['cnt'] < 120:
            logger.info("신규 종목 %s 일봉 백필...", code)
            price_collector.collect_daily(code, days=600)
            backfill_count += 1

    names = {s['code']: s['name'] for s in parsed}
    for code in top_codes:
        if code not in ALL_STOCKS and code in names:
            ALL_STOCKS[code] = names[code]

    # 로그
    top5 = [s for s in parsed if s['momentum'] > 0][:5]
    top5_str = ', '.join(
        f"{s['name']}({s['rate']:+.1f}%,금액{s['amount']/1e8:.0f}억)"
        for s in top5
    )
    logger.info("스캔 완료: %d종목 (모멘텀+%d, 감시%d) | TOP: %s",
                len(top_codes),
                sum(1 for s in parsed if s['momentum'] > 0),
                sum(1 for s in parsed if s['momentum'] <= 0 and s['amount'] > 5e9),
                top5_str)
    return top_codes


def cmd_backfill(args):
    """과거 데이터 백필."""
    logger.info("=== 백필 시작 (%d일) ===", args.days)
    api = get_api()
    db = get_db()

    from collector import CollectorScheduler
    scheduler = CollectorScheduler(api, db, ALL_STOCKS)
    stats = scheduler.run_backfill(days=args.days)

    print("\n=== 백필 결과 ===")
    for table, count in stats.items():
        print(f"  {table}: {count:,} rows")


def cmd_features(args):
    """피처 계산."""
    logger.info("=== 피처 계산 시작 ===")
    db = get_db()
    from features import FeatureEngine

    engine = FeatureEngine(db)
    total = 0
    codes = list(ALL_STOCKS.keys())

    for i, code in enumerate(codes):
        sector = SECTOR_MAP.get(code, "")
        n = engine.compute_and_save(code, sector_code=sector)
        total += n
        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{len(codes)} 종목 ({total:,} 피처)")

    print(f"\n=== 피처 계산 완료: {total:,}개 ===")


def cmd_train(args):
    """모델 학습."""
    logger.info("=== 모델 학습 시작 ===")
    db = get_db()

    from features import FeatureEngine
    from features.normalizer import FeatureNormalizer
    from models import LGBMModel, LSTMModel, Evaluator
    from models.trainer import Trainer

    engine = FeatureEngine(db)
    lgbm = LGBMModel()
    lstm = LSTMModel()
    trainer = Trainer(db, engine, lgbm, lstm)

    # 데이터 준비
    codes = list(ALL_STOCKS.keys())
    data = trainer.prepare_data(codes)
    if data is None:
        print("학습 데이터가 부족합니다. --backfill과 --features를 먼저 실행하세요.")
        return

    # 정규화
    normalizer = FeatureNormalizer()
    feature_dicts = [dict(zip(data["feature_names"], row)) for row in data["X"]]
    normalized = normalizer.fit_transform(feature_dicts)
    import numpy as np
    data["X"] = np.array([[d.get(k, 0) for k in data["feature_names"]] for d in normalized],
                          dtype=np.float32)

    # 시퀀스 재생성
    data["sequences"] = trainer._make_sequences(data["X"], SEQUENCE_LENGTH)

    # Walk-forward 학습
    if args.walk_forward:
        results = trainer.walk_forward_train(data)
        print("\n=== Walk-Forward 결과 ===")
        for r in results:
            print(f"  Fold {r['fold']}: LGBM acc={r['lgbm'].get('val_accuracy', 0):.4f}, "
                  f"time={r['training_time']:.1f}s")
    else:
        # 최종 학습
        result = trainer.train_final(data)
        print(f"\n=== 최종 학습 결과 ===")
        print(f"  LGBM: {result['lgbm']}")
        print(f"  LSTM: {result.get('lstm', {})}")
        print(f"  학습 시간: {result['training_time']:.1f}s")

    # 모델 저장
    lgbm_path = str(MODEL_DIR / "lgbm_latest.pkl")
    lstm_path = str(MODEL_DIR / "lstm_latest.pt")
    norm_path = str(MODEL_DIR / "normalizer.json")

    lgbm.save(lgbm_path)
    lstm.save(lstm_path)
    normalizer.save(norm_path)

    # 학습 로그
    db.insert_training_log({
        "model_name": "ensemble",
        "train_start_date": data["labels"][0][1] if data["labels"] else "",
        "train_end_date": data["labels"][-1][1] if data["labels"] else "",
        "metrics": result if not args.walk_forward else {"walk_forward": len(results)},
        "model_path": lgbm_path,
        "feature_importance": lgbm.feature_importance_ or {},
        "training_time_sec": result.get("training_time", 0) if not args.walk_forward else 0,
    })

    # 피처 중요도 출력
    top_feat = lgbm.get_top_features(15)
    if top_feat:
        print("\n=== Top 15 피처 ===")
        for name, imp in top_feat:
            print(f"  {name}: {imp:.0f}")


def _save_nxt_prices(api, db, watchlist, today):
    """NXT 가격을 DB에 저장 — 내일 예측의 핵심 입력 데이터.

    워치리스트 전체 + 보유 종목의 NXT 가격을 수집하여
    nxt_prices 테이블에 기록. 이전에는 실시간으로만 쓰고 버렸음.
    """
    current_time = datetime.now().strftime("%H:%M")
    saved = 0

    # 당일 종가 미리 수집
    close_map = {}
    rows = db.execute_raw(
        "SELECT stock_code, close FROM daily_prices WHERE date=?", (today,))
    for r in rows:
        close_map[r['stock_code']] = r['close']

    for code in watchlist:
        try:
            multi = api.get_multi_market_price(code)
            nxt = multi.get('nxt', {})
            if not nxt.get('price'):
                continue

            nxt_price = nxt['price']
            krx_close = close_map.get(code, 0)
            vs_close = (nxt_price - krx_close) / krx_close * 100 if krx_close else 0

            conn = db._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO nxt_prices
                       (stock_code, date, time, nxt_price, krx_close,
                        nxt_vs_close_pct, spread_pct, volume)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (code, today, current_time, nxt_price, krx_close,
                     vs_close, multi.get('spread_pct', 0),
                     nxt.get('volume', 0))
                )
                conn.commit()
            finally:
                conn.close()
            saved += 1
        except Exception:
            pass

    logger.info("[NXT DB] %d종목 가격 저장 (%s)", saved, current_time)
    return saved


def _run_nxt_monitor(api, db, engine, ensemble, normalizer, paper, today):
    """NXT 야간거래 모니터링 — 보유 종목 손절/익절 + 기회 포착.

    NXT는 유동성이 낮아서:
    - 보유 종목의 NXT 가격으로 손절/익절 판단
    - ML 상승 예측 + NXT 가격이 장중 종가보다 낮으면 매수 기회
    - NXT↔KRX 스프레드가 크면 차익 기회
    """
    import numpy as np

    if not paper.portfolio.positions:
        logger.info("[NXT] 보유 종목 없음 — 스킵")
        return

    # NXT 가격 수집 (보유 종목)
    nxt_prices = {}
    spreads = {}
    for code in list(paper.portfolio.positions.keys()):
        try:
            multi = api.get_multi_market_price(code)
            if multi.get('nxt') and multi['nxt'].get('price'):
                nxt_prices[code] = multi['nxt']['price']
                spreads[code] = multi.get('spread_pct', 0)
        except Exception:
            pass

    if not nxt_prices:
        logger.info("[NXT] 가격 수집 실패")
        return

    sell_count = 0
    for code, nxt_price in nxt_prices.items():
        pos = paper.portfolio.positions.get(code)
        if not pos:
            continue

        pnl = (nxt_price - pos.avg_price) / pos.avg_price * 100
        name = ALL_STOCKS.get(code, code)
        spread = spreads.get(code, 0)

        # NXT 긴급 손절 (-5%, 절대 안전장치)
        if pnl < -5.0:
            paper._sell(code, nxt_price,
                       reason=f"NXT 긴급 손절 {pnl:.1f}%",
                       now=datetime.now().strftime("%Y%m%d %H:%M:%S"))
            logger.info("[NXT] 긴급 손절: %s %+.1f%% (NXT가 %s원)", name, pnl, f"{nxt_price:,}")
            sell_count += 1

        # NXT 트레일링 스톱 (고점 대비 -5%, 수익 보호)
        elif pnl > 0 and pos.highest_price > 0:
            from_high = (nxt_price - pos.highest_price) / pos.highest_price * 100
            if from_high < -5.0:
                paper._sell(code, nxt_price,
                           reason=f"NXT 트레일링 (고점 대비 {from_high:.1f}%)",
                           now=datetime.now().strftime("%Y%m%d %H:%M:%S"))
                logger.info("[NXT] 트레일링: %s %+.1f%% (NXT가 %s원)", name, pnl, f"{nxt_price:,}")
                sell_count += 1

    # NXT 상태 로그
    held_str = ', '.join(
        f"{ALL_STOCKS.get(c,c)}({nxt_prices[c]:,}원,{(nxt_prices[c]-paper.portfolio.positions[c].avg_price)/paper.portfolio.positions[c].avg_price*100:+.1f}%)"
        for c in nxt_prices if c in paper.portfolio.positions
    )
    logger.info("[NXT] 보유 %d종목 | 매도 %d | %s",
                len(nxt_prices), sell_count, held_str[:200])

    # NXT 가격으로 포트폴리오 갱신
    paper.update_prices(nxt_prices)


def _auto_forecast_tomorrow(api, db, engine, ensemble, normalizer, watchlist, today):
    """NXT 종료 후 내일 갭업 예측 자동 생성.

    NXT 가격 + 당일 수급/기술적 분석을 종합하여
    내일 갭업 가능성이 높은 종목 TOP 5를 predictions에 기록.

    핵심 교훈 반영:
    - 거래량비 >= 1.0 필터 (0.5x 미만은 모멘텀 부족)
    - NXT 추가상승 + 수급 복합 판단
    - 단일 팩터 과신 금지
    """
    import numpy as np
    from nn_config import CLASS_NAMES

    # 1) 당일 종가 + 거래량 + 수급
    day_data = db.execute_raw(
        "SELECT stock_code, close, volume, change_rate FROM daily_prices WHERE date=?",
        (today,))
    close_map = {r['stock_code']: r for r in day_data}

    # 20일 평균 거래량
    avg_vols = {}
    for code in watchlist:
        rows = db.execute_raw(
            "SELECT AVG(volume) as avg_vol FROM daily_prices "
            "WHERE stock_code=? AND date<? ORDER BY date DESC LIMIT 20",
            (code, today))
        if rows and rows[0]['avg_vol']:
            avg_vols[code] = rows[0]['avg_vol']

    # 2) NXT 가격 (오늘 최신)
    nxt_data = db.execute_raw(
        "SELECT stock_code, nxt_price, nxt_vs_close_pct "
        "FROM nxt_prices WHERE date=? "
        "ORDER BY time DESC", (today,))
    nxt_latest = {}
    for r in nxt_data:
        if r['stock_code'] not in nxt_latest:  # 가장 최근 시간만
            nxt_latest[r['stock_code']] = r

    # 3) 수급 (외국인/기관)
    flow_data = db.execute_raw(
        "SELECT stock_code, foreign_net, institution_net "
        "FROM investor_flow WHERE date=?", (today,))
    flow_map = {r['stock_code']: r for r in flow_data}

    # 4) 종합 점수 계산
    candidates = []
    for code in watchlist:
        d = close_map.get(code)
        if not d or not d['close']:
            continue

        score = 0
        reasons = []

        # 당일 상승률
        rate = d.get('change_rate', 0) or 0
        if rate > 3:
            score += 15
            reasons.append(f"당일+{rate:.1f}%")

        # 거래량비 (핵심 필터!)
        vol = d.get('volume', 0) or 0
        avg_vol = avg_vols.get(code, 0)
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        if vol_ratio < 1.0:
            continue  # 거래량비 < 1.0 → 스킵 (교훈 반영)
        if vol_ratio >= 2.0:
            score += 25
            reasons.append(f"거래량{vol_ratio:.1f}x")
        elif vol_ratio >= 1.5:
            score += 15
            reasons.append(f"거래량{vol_ratio:.1f}x")
        else:
            score += 5

        # NXT 추가 상승
        nxt = nxt_latest.get(code)
        if nxt and nxt['nxt_vs_close_pct']:
            nxt_chg = nxt['nxt_vs_close_pct']
            if nxt_chg >= 3.0:
                score += 30
                reasons.append(f"NXT+{nxt_chg:.1f}%")
            elif nxt_chg >= 1.0:
                score += 15
                reasons.append(f"NXT+{nxt_chg:.1f}%")

        # 외국인+기관 쌍끌이
        flow = flow_map.get(code, {})
        f_net = flow.get('foreign_net', 0) or 0
        i_net = flow.get('institution_net', 0) or 0
        if f_net > 0 and i_net > 0:
            score += 20
            reasons.append("외+기 쌍끌이")
        elif f_net > 100000 or i_net > 100000:
            score += 10
            reasons.append(f"{'외' if f_net > i_net else '기'}관 매수")

        if score >= 40:
            candidates.append({
                'code': code,
                'score': score,
                'reasons': reasons,
                'rate': rate,
                'vol_ratio': vol_ratio,
            })

    candidates.sort(key=lambda x: -x['score'])
    top = candidates[:5]

    if not top:
        logger.info("[예측] 내일 갭업 후보 없음 (조건 미달)")
        return

    # 5) predictions 테이블에 기록
    for c in top:
        confidence = min(0.85, c['score'] / 100)
        pred_class = 2  # 상승
        db.insert_prediction({
            'stock_code': c['code'],
            'date': today,
            'model_name': 'auto_forecast',
            'predicted_class': pred_class,
            'class_probabilities': [0, int((1-confidence)*100), int(confidence*100)],
            'confidence': confidence,
        })
        name = ALL_STOCKS.get(c['code'], c['code'])
        logger.info("[예측] %s(%s) score=%d conf=%.0f%% | %s",
                    name, c['code'], c['score'], confidence*100,
                    ', '.join(c['reasons']))

    # 텔레그램으로도 전송
    try:
        from telegram_bot import send_message
        lines = ["[내일 갭업 예측 TOP 5]"]
        for i, c in enumerate(top, 1):
            name = ALL_STOCKS.get(c['code'], c['code'])
            conf = min(85, c['score'])
            lines.append(f"{i}. {name}({c['code']}) "
                         f"신뢰도 {conf}% | {', '.join(c['reasons'])}")
        lines.append("\n* 거래량비 >= 1.0 필터 적용")
        send_message('\n'.join(lines))
    except Exception:
        pass


def _auto_verify_predictions(db):
    """어제까지의 미검증 예측을 자동 검증 + 후회 기록.

    16:40에 실행 — 장후 데이터 수집(16:30) 완료 후.
    """
    from nn_config import CLASS_NAMES

    unverified = db.execute_raw(
        """SELECT id, stock_code, date, model_name, predicted_class, confidence
           FROM predictions
           WHERE actual_return IS NULL
           AND date < strftime('%Y%m%d', 'now', 'localtime')
           ORDER BY date ASC""")

    if not unverified:
        logger.info("[검증] 미검증 예측 없음")
        return

    verified = 0
    regrets = 0

    for pred in unverified:
        code = pred['stock_code']
        pred_date = pred['date']

        base = db.execute_raw(
            "SELECT close FROM daily_prices WHERE stock_code=? AND date=?",
            (code, pred_date))
        if not base or not base[0]['close']:
            continue

        next_days = db.execute_raw(
            "SELECT date, open, high, close FROM daily_prices "
            "WHERE stock_code=? AND date>? ORDER BY date LIMIT 5",
            (code, pred_date))
        if not next_days:
            continue

        base_close = base[0]['close']
        nd = next_days[0]
        close_pct = (nd['close'] - base_close) / base_close * 100 if nd['close'] else 0

        # 5일 수익률 (있으면)
        if len(next_days) >= 5:
            ret = (next_days[4]['close'] - base_close) / base_close * 100
        else:
            ret = close_pct

        # 3-class: 0=하락, 1=보합, 2=상승
        if ret < -1.5:
            actual_class = 0  # 하락
        elif ret > 1.5:
            actual_class = 2  # 상승
        else:
            actual_class = 1  # 보합

        db.update_prediction_actual(pred['id'], actual_class, ret)
        verified += 1

        # 상승 예측 → 하락 = 후회
        pred_class = pred['predicted_class']
        if pred_class == 2 and ret < -1:
            name = ALL_STOCKS.get(code, code)
            db.insert_regret({
                'trade_id': 0,
                'stock_code': code,
                'trade_date': pred_date,
                'action_taken': f'[{pred["model_name"]}] {CLASS_NAMES[pred_class]} 예측',
                'optimal_action': '관망',
                'regret_score': min(1.0, abs(ret) / 10),
                'return_if_optimal': 0,
                'return_actual': ret,
                'lesson': f'[자동검증] {name} {CLASS_NAMES[pred_class]} → 실제 {ret:+.1f}% ({CLASS_NAMES[actual_class]})',
            })
            regrets += 1

    logger.info("[검증] %d건 검증, %d건 후회 기록", verified, regrets)

    # 텔레그램 요약
    if verified > 0:
        try:
            from telegram_bot import send_message
            # 최근 forecast/auto_forecast 결과만 요약
            recent = db.execute_raw(
                """SELECT stock_code, model_name, predicted_class, actual_class,
                          actual_return, confidence
                   FROM predictions
                   WHERE model_name IN ('human_forecast', 'auto_forecast')
                   AND actual_return IS NOT NULL
                   ORDER BY id DESC LIMIT 10""")
            if recent:
                lines = [f"[예측 검증 결과] {verified}건"]
                for r in recent:
                    name = ALL_STOCKS.get(r['stock_code'], r['stock_code'])
                    hit = "O" if r['predicted_class'] == r['actual_class'] else "X"
                    lines.append(f"  {hit} {name}: "
                                 f"{CLASS_NAMES[r['predicted_class']]}→"
                                 f"{CLASS_NAMES[r['actual_class']]} "
                                 f"({r['actual_return']:+.1f}%)")
                send_message('\n'.join(lines))
        except Exception:
            pass


def _run_predict_and_trade(api, db, engine, ensemble, normalizer,
                           paper, watchlist, today):
    """ML 중심 매매 — 신경망이 판단, 시그널은 참고.

    매수 조건:
    - ML이 상승 예측 (predicted_class == 2)
    - 확신도 기반 포지션 사이징
    - 거래량/호가는 ML 피처로 간접 반영
    - 이미 급등한 종목(+15%) 제외 (고점 추격 방지)

    매도 조건 (ML 기반):
    1. ML 하락 예측 → 즉시 매도
    2. ML 보합 + 수익 중 → 익절 (모멘텀 끝)
    3. ML 보합 + 손실 3일↑ → 청산 (기회비용)
    4. 긴급 손절 -5% (안전장치)
    5. 트레일링 스톱 -5% (수익 보호)
    """
    import numpy as np
    global _latest_momentum, _prev_momentum, _us_sentiment

    # 현재가: 거래대금 API에서 이미 받은 bulk 데이터 활용 (API콜 최소화)
    price_map = {}
    for code, data in _latest_momentum.items():
        if data.get('price'):
            price_map[code] = data['price']
    # 보유 종목 중 bulk에 없는 것만 개별 API
    for code in list(paper.portfolio.positions.keys()):
        if code not in price_map:
            try:
                info = api.get_current_price(code)
                if info.get("price"):
                    price_map[code] = info["price"]
            except Exception:
                pass
    logger.info("가격 수집: bulk %d + 개별 %d = %d종목",
                sum(1 for c in price_map if c in _latest_momentum),
                sum(1 for c in price_map if c not in _latest_momentum),
                len(price_map))

    # ML 예측
    predictions = {}
    feature_names = normalizer.get_feature_names()

    pred_count = 0
    for code in watchlist:
        try:
            sector = SECTOR_MAP.get(code, "")
            feat = engine.compute_features_for_stock(code, today, sector)
            if not feat:
                continue

            # 실시간 시그널을 피처로 주입 (모델이 학습하도록)
            mom = _latest_momentum.get(code, {})
            vol_surge = detect_volume_surge(db, code, mom.get('volume', 0))
            ob_ratio = detect_orderbook_imbalance(db, code)
            feat['realtime_vol_surge'] = vol_surge
            feat['realtime_ob_ratio'] = ob_ratio
            feat['realtime_rate'] = mom.get('rate', 0)

            norm_feat = normalizer.transform(feat)
            x_flat = np.array([[norm_feat.get(k, 0) for k in feature_names]],
                              dtype=np.float32)

            # 실제 20일 시퀀스 구성 (DB에서 과거 피처 로드)
            seq_data = engine.get_feature_sequence(code, today, feature_names, SEQUENCE_LENGTH)
            if seq_data:
                # 정규화 적용
                norm_seq = []
                for day_vec_dict in [dict(zip(feature_names, day)) for day in seq_data]:
                    nd = normalizer.transform(day_vec_dict)
                    norm_seq.append([nd.get(k, 0) for k in feature_names])
                x_seq = np.array([norm_seq], dtype=np.float32)
            else:
                # 피처 없으면 폴백 (오늘 데이터만)
                x_seq = np.zeros((1, SEQUENCE_LENGTH, len(feature_names)), dtype=np.float32)
                x_seq[0, -1, :] = x_flat[0]

            result = ensemble.predict_single(x_flat, x_seq)
            result["stock_code"] = code
            predictions[code] = result
            pred_count += 1

            db.insert_prediction({
                "stock_code": code,
                "date": today,
                "model_name": "ensemble",
                "predicted_class": result["predicted_class"],
                "class_probabilities": result["class_probabilities"],
                "confidence": result["confidence"],
            })
        except Exception as e:
            logger.error("%s 예측 실패: %s", code, e)

    logger.info("ML 예측 완료: %d/%d종목", pred_count, len(watchlist))

    # === ML 중심 매매 시그널 ===
    buy_candidates = []
    sell_signals = []
    held_momenta = {}

    for code in watchlist:
        mom = _latest_momentum.get(code, {})
        prev = _prev_momentum.get(code, {})
        ml = predictions.get(code, {})
        price = price_map.get(code, 0)
        if not price:
            continue

        momentum_score = mom.get('momentum', 0)
        prev_momentum = prev.get('momentum', 0)
        rate = mom.get('rate', 0)
        amount = mom.get('amount', 0)
        ml_class = ml.get('predicted_class', 2)
        name = ALL_STOCKS.get(code, code)
        momentum_accel = momentum_score - prev_momentum if prev_momentum > 0 else 0

        # === 보유 종목 판단 ===
        if code in paper.portfolio.positions:
            pos = paper.portfolio.positions[code]
            pnl = (price - pos.avg_price) / pos.avg_price * 100
            ml_conf = ml.get('confidence', 0)
            held_momenta[code] = {'momentum': momentum_score, 'rate': rate, 'pnl': pnl}

            should_sell = False
            reason = ""
            trailing_drop = (price - pos.highest_price) / pos.highest_price * 100 if pos.highest_price > 0 else 0

            # 1. 긴급 손절 -5% (절대 안전장치)
            if pnl < -5.0:
                should_sell = True
                reason = f"긴급 손절 {pnl:.1f}%"
            # 2. 트레일링 스톱: 고점 대비 -5% (수익 보호)
            elif trailing_drop < -5.0 and pnl > 0:
                should_sell = True
                reason = f"트레일링 스톱 (고점 대비 {trailing_drop:.1f}%, 수익 {pnl:.1f}%)"
            # 3. ML이 하락(0) 예측 → 매도
            elif ml_class == 0 and ml_conf > 0.4:
                should_sell = True
                reason = f"ML 하락 (확신 {ml_conf:.0%}, pnl {pnl:+.1f}%)"
            # 4. ML이 보합(1) + 수익 중 → 모멘텀 끝, 익절
            elif ml_class == 1 and ml_conf > 0.6 and pnl > 1.0:
                should_sell = True
                reason = f"ML 보합→익절 (확신 {ml_conf:.0%}, pnl {pnl:+.1f}%)"
            # 5. ML이 보합(1) + 3일 이상 손실 → 기회비용, 청산
            elif ml_class == 1 and ml_conf > 0.5 and pnl < -1.0:
                entry_date = getattr(pos, 'entry_date', '')
                if entry_date:
                    try:
                        from datetime import datetime as dt
                        days_held = (dt.now() - dt.strptime(entry_date, "%Y%m%d")).days
                        if days_held >= 3:
                            should_sell = True
                            reason = f"ML 보합+손실→청산 (보유{days_held}일, pnl {pnl:+.1f}%)"
                    except (ValueError, AttributeError):
                        pass

            if should_sell:
                sell_signals.append({
                    'stock_code': code, 'predicted_class': ml_class,
                    'confidence': max(ml_conf, 0.8), 'reason': reason,
                })
                logger.info("매도: %s — %s (수익 %+.1f%%)", name, reason, pnl)
            continue

        # === ML 중심 매수 판단 ===
        ml_conf = ml.get('confidence', 0)

        # ML이 상승(2) 예측한 종목만 매수 후보
        if ml_class == 2 and ml_conf > 0.4:
            # 이미 급등한 종목은 제외 (고점 추격 방지)
            if rate > 15.0:
                logger.info("고점 제외: %s (%+.1f%% 이미 급등)", name, rate)
            # KOSPI 급락 시 매수 자제 (US 센티먼트 -0.5 이하, 확신 0.7 미만)
            elif _us_sentiment < -0.5 and ml_conf < 0.7:
                pass  # 고확신만 허용
            else:
                buy_candidates.append({
                    'stock_code': code,
                    'predicted_class': ml_class,
                    'confidence': ml_conf,
                    'momentum': momentum_score,
                    'rate': rate,
                    'name': name,
                    'amount': amount,
                })

    # ML 확신도 높은 순으로 정렬
    buy_candidates.sort(key=lambda x: -x['confidence'])

    # === 포지션 회전: ML이 매수 확신 높은 종목 vs 보유 중 손실 종목 ===
    from nn_config import MAX_POSITIONS
    num_held = len(paper.portfolio.positions)
    free_slots = MAX_POSITIONS - num_held + len(sell_signals)

    cash_ratio = paper.portfolio.cash / max(paper.portfolio.total_value, 1)
    need_rotation = (free_slots <= 0 or cash_ratio < 0.05) and buy_candidates
    logger.info("포지션: 보유 %d, 슬롯 %d, 현금 %.1f%%, ML매수후보 %d",
                num_held, free_slots, cash_ratio * 100, len(buy_candidates))

    if need_rotation and held_momenta:
        # 손실 종목 우선 교체 (pnl 낮은 순)
        weakest = sorted(held_momenta.items(), key=lambda x: x[1]['pnl'])
        best_buy = buy_candidates[0]

        for weak_code, weak_data in weakest:
            if not buy_candidates:
                break
            weak_pnl = weak_data.get('pnl', 0)
            best_buy = buy_candidates[0]

            # 교체 조건: 보유종목 손실 중 + ML이 상승 확신
            should_rotate = False
            if weak_pnl < -1.0 and best_buy['confidence'] > 0.5:
                should_rotate = True
            elif weak_pnl < 0.5 and best_buy['confidence'] > 0.7:
                should_rotate = True

            if should_rotate:
                weak_name = ALL_STOCKS.get(weak_code, weak_code)
                if weak_code in price_map:
                    paper._sell(weak_code, price_map[weak_code],
                               reason=f"교체→{best_buy['name']}(ML확신{best_buy['confidence']:.0%})",
                               now=datetime.now().strftime("%Y%m%d %H:%M:%S"))
                    free_slots += 1
                    logger.info("교체: %s(pnl=%+.1f%%) → %s(ML %s, 확신%.0f%%)",
                               weak_name, weak_pnl,
                               best_buy['name'],
                               CLASS_NAMES[best_buy['predicted_class']],
                               best_buy['confidence'] * 100)
                if free_slots >= 3:
                    break

    # 매도 먼저 → 매수
    if sell_signals:
        paper.execute_signals(sell_signals, price_map)
    if buy_candidates:
        paper.execute_signals(buy_candidates, price_map)

    # 현재가 갱신 + 스냅샷
    paper.update_prices(price_map)
    paper.daily_close()

    # 이전 모멘텀 저장 (다음 주기 비교용)
    _prev_momentum = {k: v.copy() for k, v in _latest_momentum.items()}

    # 로그
    CLASS_NAMES = ['하락','보합','상승']
    buy_str = ', '.join(
        f"{s['name']}({CLASS_NAMES[s['predicted_class']]},{s['confidence']:.0%})"
        for s in buy_candidates[:5]
    )
    sell_str = ', '.join(
        f"{ALL_STOCKS.get(s['stock_code'], s['stock_code'])}"
        for s in sell_signals
    )
    logger.info("매수 %d [%s] 매도 %d [%s] 총평가 %s원",
                len(buy_candidates), buy_str,
                len(sell_signals), sell_str,
                f"{paper.portfolio.total_value:,}")


def cmd_paper(args):
    """모의투자 시작 (메인 루프)."""
    logger.info("=== 모의투자 시작 ===")
    api = get_api()
    db = get_db()

    from features import FeatureEngine
    from features.normalizer import FeatureNormalizer
    from models import LGBMModel, LSTMModel, EnsembleModel
    from trading import Portfolio, PaperEngine
    from collector import CollectorScheduler
    import numpy as np

    # 모델 로드
    lgbm = LGBMModel()
    lstm = LSTMModel()

    lgbm_path = str(MODEL_DIR / "lgbm_latest.pkl")
    lstm_path = str(MODEL_DIR / "lstm_latest.pt")
    norm_path = str(MODEL_DIR / "normalizer.json")

    try:
        lgbm.load(lgbm_path)
    except FileNotFoundError:
        print("모델 파일이 없습니다. --train을 먼저 실행하세요.")
        return

    try:
        lstm.load(lstm_path)
    except Exception:
        logger.warning("LSTM 모델 로드 실패 — LightGBM만 사용")

    normalizer = FeatureNormalizer()
    try:
        normalizer.load(norm_path)
    except FileNotFoundError:
        logger.warning("Normalizer 없음")

    ensemble = EnsembleModel(lgbm, lstm)
    engine = FeatureEngine(db)
    collector = CollectorScheduler(api, db, ALL_STOCKS)

    # 포트폴리오 복원 또는 신규 생성
    portfolio = Portfolio(PAPER_INITIAL_CAPITAL, trade_type="paper")
    latest = db.get_latest_portfolio("paper")
    if latest:
        portfolio.load_from_snapshot(latest)
        logger.info("포트폴리오 복원: 총 %s원", f"{portfolio.total_value:,}")

    paper = PaperEngine(db, portfolio)

    print(f"모의투자 시작: 자본금 {portfolio.total_value:,}원")
    print("Ctrl+C로 중단\n")

    # 메인 루프
    last_collect = ""
    last_predict = ""
    last_report = ""
    dynamic_watchlist = list(ALL_STOCKS.keys())[:21]  # 초기값

    while True:
        try:
            now = datetime.now()
            today = now.strftime("%Y%m%d")
            current_time = now.strftime("%H:%M")

            # 평일만
            if now.weekday() >= 5:
                time.sleep(60)
                continue

            # === 장전 수집 + US 센티먼트 (08:00) ===
            if current_time == "08:00" and last_collect != f"{today}_pre":
                logger.info("[08:00] 장전 수집 + US 센티먼트")
                collector.run_pre_market()
                check_us_sentiment(db)
                last_collect = f"{today}_pre"

            # === 장중 10분마다 예측+매매 (09:10, 09:20, ..., 15:20) ===
            if "09:10" <= current_time <= "15:20":
                mm = int(current_time[3:5])
                if mm % 10 == 0:
                    intraday_key = f"{today}_{current_time[:5]}"
                    if last_predict != intraday_key:
                        logger.info("[%s] 10분 주기 예측+매매", current_time)

                        # 모멘텀 TOP 갱신 (매 주기마다)
                        try:
                            dynamic_watchlist = get_dynamic_watchlist(
                                api, db, portfolio)
                        except Exception as e:
                            logger.error("워치리스트 갱신 실패: %s", e)

                        # 예측 + 매매
                        _run_predict_and_trade(
                            api, db, engine, ensemble, normalizer,
                            paper, dynamic_watchlist, today,
                        )
                        last_predict = intraday_key

            # === 장후 수집 + 마감 (16:30) ===
            if current_time == "16:30" and last_predict != f"{today}_close":
                logger.info("[16:30] 장후 처리")
                collector.run_post_market()

                try:
                    dynamic_watchlist = get_dynamic_watchlist(
                        api, db, portfolio)
                except Exception as e:
                    logger.error("워치리스트 갱신 실패: %s", e)

                _run_predict_and_trade(
                    api, db, engine, ensemble, normalizer,
                    paper, dynamic_watchlist, today,
                )

                # 일일 마감 + 후회 분석
                paper.daily_close()
                paper.analyze_regret()
                last_predict = f"{today}_close"

            # === NXT 야간거래 모니터링 (18:00~20:00, 30분 주기) ===
            if "18:00" <= current_time <= "20:00":
                mm = int(current_time[3:5])
                if mm % 30 == 0:
                    nxt_key = f"{today}_nxt_{current_time[:5]}"
                    if last_predict != nxt_key:
                        logger.info("[NXT %s] 야간거래 모니터링", current_time)
                        # NXT 가격 DB 저장 (워치리스트 전체)
                        try:
                            _save_nxt_prices(api, db, dynamic_watchlist, today)
                        except Exception as e:
                            logger.error("[NXT] 가격 저장 실패: %s", e)
                        _run_nxt_monitor(
                            api, db, engine, ensemble, normalizer,
                            paper, today,
                        )
                        last_predict = nxt_key

            # === NXT 종료 후 내일 예측 자동 기록 (20:30) ===
            if current_time == "20:30" and last_collect != f"{today}_forecast":
                logger.info("[20:30] 내일 예측 자동 생성 (NXT 데이터 기반)")
                try:
                    _auto_forecast_tomorrow(api, db, engine, ensemble,
                                            normalizer, dynamic_watchlist, today)
                except Exception as e:
                    logger.error("내일 예측 실패: %s", e)
                last_collect = f"{today}_forecast"

            # === 어제 예측 자동 검증 (16:40, 장후 수집 완료 후) ===
            if current_time == "16:40" and last_collect != f"{today}_verify":
                logger.info("[16:40] 어제 예측 자동 검증")
                try:
                    _auto_verify_predictions(db)
                except Exception as e:
                    logger.error("예측 검증 실패: %s", e)
                last_collect = f"{today}_verify"

            # === 장중 온라인 학습 (12:00 오전 데이터 반영) ===
            if current_time == "12:00" and last_collect != f"{today}_online":
                logger.info("[12:00] 장중 온라인 학습")
                try:
                    from features import FeatureEngine as FE
                    from models.trainer import Trainer
                    import numpy as np

                    ol_engine = FE(db)
                    trainer = Trainer(db, ol_engine, lgbm, lstm)
                    codes = list(ALL_STOCKS.keys())
                    data = trainer.prepare_data(codes)
                    if data is not None:
                        feat_dicts = [dict(zip(data["feature_names"], row)) for row in data["X"]]
                        normed = [normalizer.transform(d) for d in feat_dicts]
                        X = np.array(
                            [[d.get(k, 0) for k in data["feature_names"]] for d in normed],
                            dtype=np.float32)
                        weights = trainer._compute_regret_weights(data["labels"])
                        result = lgbm.incremental_train(X, data["y"],
                                                         sample_weight=weights,
                                                         n_new_trees=50)
                        lgbm.save(str(MODEL_DIR / "lgbm_latest.pkl"))
                        ensemble = EnsembleModel(lgbm, lstm)
                        logger.info("온라인 학습: %s", result)
                except Exception as e:
                    logger.error("온라인 학습 실패: %s", e)
                last_collect = f"{today}_online"

            # === 일일 전체 재학습 (17:30 장마감 후) ===
            if current_time == "17:30" and last_collect != f"{today}_retrain":
                logger.info("[17:30] 일일 전체 재학습 시작")
                try:
                    from features import FeatureEngine as FE
                    from features.normalizer import FeatureNormalizer as FN
                    from models.trainer import Trainer
                    import numpy as np

                    retrain_engine = FE(db)
                    retrain_normalizer = FN()
                    trainer = Trainer(db, retrain_engine, lgbm, lstm)

                    codes = list(ALL_STOCKS.keys())
                    data = trainer.prepare_data(codes)
                    if data is not None:
                        feat_dicts = [dict(zip(data["feature_names"], row)) for row in data["X"]]
                        normed = retrain_normalizer.fit_transform(feat_dicts)
                        data["X"] = np.array(
                            [[d.get(k, 0) for k in data["feature_names"]] for d in normed],
                            dtype=np.float32)
                        data["sequences"] = trainer._make_sequences(data["X"], SEQUENCE_LENGTH)

                        result = trainer.train_final(data)
                        lgbm.save(str(MODEL_DIR / "lgbm_latest.pkl"))
                        lstm.save(str(MODEL_DIR / "lstm_latest.pt"))
                        retrain_normalizer.save(str(MODEL_DIR / "normalizer.json"))
                        normalizer = retrain_normalizer
                        ensemble = EnsembleModel(lgbm, lstm)
                        logger.info("일일 재학습 완료: %s", result)
                except Exception as e:
                    logger.error("재학습 실패: %s", e, exc_info=True)
                last_collect = f"{today}_retrain"

            # 텔레그램 리포트 (20:00)
            if current_time == "20:00" and last_report != today:
                report = paper.generate_report()
                try:
                    from telegram_bot import send_message
                    send_message(report)
                except Exception as e:
                    logger.error("텔레그램 전송 실패: %s", e)
                print(report)
                last_report = today

            # 장중 호가 수집 (매분)
            if "09:00" <= current_time <= "15:30":
                now_ts = time.time()
                if now_ts - collector._last_orderbook >= 60:
                    collector._last_orderbook = now_ts
                    collector.orderbook.collect_all_snapshots(dynamic_watchlist)

            time.sleep(20)  # 20초 간격으로 시간 체크

        except KeyboardInterrupt:
            logger.info("모의투자 중단")
            paper.daily_close()
            print("\n모의투자 중단됨.")
            break
        except Exception as e:
            logger.error("메인 루프 에러: %s", e, exc_info=True)
            time.sleep(60)


def cmd_predict(args):
    """즉시 예측 + 매매 실행 (테스트용)."""
    logger.info("=== 즉시 예측 시작 ===")
    api = get_api()
    db = get_db()

    from features import FeatureEngine
    from features.normalizer import FeatureNormalizer
    from models import LGBMModel, LSTMModel, EnsembleModel
    from trading import Portfolio, PaperEngine
    import numpy as np

    engine = FeatureEngine(db)
    lgbm = LGBMModel()
    lstm = LSTMModel()
    normalizer = FeatureNormalizer()

    lgbm.load(str(MODEL_DIR / "lgbm_latest.pkl"))
    try:
        lstm.load(str(MODEL_DIR / "lstm_latest.pt"))
    except Exception:
        logger.warning("LSTM 로드 실패 — LightGBM만 사용")
        lstm = None
    normalizer.load(str(MODEL_DIR / "normalizer.json"))

    ensemble = EnsembleModel(lgbm, lstm)

    portfolio = Portfolio(PAPER_INITIAL_CAPITAL, trade_type="paper")
    latest = db.get_latest_portfolio("paper")
    if latest:
        portfolio.load_from_snapshot(latest)

    paper = PaperEngine(db, portfolio)

    today = datetime.now().strftime("%Y%m%d")

    # 거래대금 TOP 종목 동적 선정
    watchlist = get_dynamic_watchlist(api, db, portfolio)
    print(f"동적 워치리스트: {len(watchlist)}종목")

    # 현재가 수집
    price_map = {}
    for code in watchlist:
        try:
            info = api.get_current_price(code)
            if info.get("price"):
                price_map[code] = info["price"]
        except Exception:
            pass
    print(f"현재가 수집: {len(price_map)}종목")

    # 피처 + 예측
    predictions = []
    feature_names = normalizer.get_feature_names()
    CLASS_NAMES = ["하락", "보합", "상승"]

    for code in watchlist:
        try:
            sector = SECTOR_MAP.get(code, "")
            feat = engine.compute_features_for_stock(code, today, sector)
            if not feat:
                continue

            norm_feat = normalizer.transform(feat)
            x_flat = np.array([[norm_feat.get(k, 0) for k in feature_names]],
                              dtype=np.float32)

            # 실제 20일 시퀀스
            seq_data = engine.get_feature_sequence(code, today, feature_names, SEQUENCE_LENGTH)
            if seq_data:
                norm_seq = []
                for day_vec_dict in [dict(zip(feature_names, day)) for day in seq_data]:
                    nd = normalizer.transform(day_vec_dict)
                    norm_seq.append([nd.get(k, 0) for k in feature_names])
                x_seq = np.array([norm_seq], dtype=np.float32)
            else:
                x_seq = np.zeros((1, SEQUENCE_LENGTH, len(feature_names)), dtype=np.float32)
                x_seq[0, -1, :] = x_flat[0]

            result = ensemble.predict_single(x_flat, x_seq)
            result["stock_code"] = code

            stock_name = ALL_STOCKS.get(code, code)
            cls = result["predicted_class"]
            conf = result["confidence"]
            price = price_map.get(code, 0)
            print(f"  {stock_name}({code}): {CLASS_NAMES[cls]} (확신도 {conf:.1%}) 현재가 {price:,}원")

            predictions.append(result)

            db.insert_prediction({
                "stock_code": code,
                "date": today,
                "model_name": "ensemble",
                "predicted_class": cls,
                "class_probabilities": result["class_probabilities"],
                "confidence": conf,
            })
        except Exception as e:
            logger.error("%s 예측 실패: %s", code, e)

    # 매매 실행
    if predictions and price_map:
        paper.execute_signals(predictions, price_map)
        paper.daily_close()

    # 후회 분석
    paper.analyze_regret()

    print(f"\n총 {len(predictions)}종목 예측 완료")
    print(f"포트폴리오: {portfolio.total_value:,.0f}원 (현금 {portfolio.cash:,.0f}원)")
    if portfolio.positions:
        print(f"보유: {len(portfolio.positions)}종목")


def cmd_status(args):
    """현재 상태 조회."""
    db = get_db()

    # DB 통계
    stats = db.get_db_stats()
    print("=== DB 통계 ===")
    for table, count in sorted(stats.items()):
        if count > 0:
            print(f"  {table}: {count:,}")

    # 포트폴리오
    latest = db.get_latest_portfolio("paper")
    if latest:
        print(f"\n=== 모의투자 포트폴리오 ===")
        print(f"  날짜: {latest['date']}")
        print(f"  총 평가: {latest['total_value']:,}원")
        print(f"  현금: {latest['cash']:,}원")
        print(f"  수익률: {latest['cumulative_return']:+.2f}%")
        print(f"  Sharpe: {latest.get('sharpe_ratio', 0):.2f}")
        print(f"  MDD: {latest.get('max_drawdown', 0):.1f}%")
        print(f"  승률: {latest.get('win_rate', 0):.1%}")

        positions = latest.get("positions", [])
        if positions:
            print(f"\n  보유 종목 ({len(positions)}개):")
            for p in positions:
                print(f"    {p['stock_name']}: {p['quantity']}주 @ {p['avg_price']:,.0f}원 "
                      f"→ {p['current_price']:,.0f}원 ({p['pnl']:+.1f}%)")
    else:
        print("\n모의투자 기록 없음")

    # 최근 후회
    regrets = db.execute_raw(
        "SELECT * FROM regret_log ORDER BY id DESC LIMIT 5"
    )
    if regrets:
        print(f"\n=== 최근 후회 ===")
        for r in regrets:
            print(f"  [{r['trade_date']}] {r['lesson']}")


def cmd_report(args):
    """텔레그램 리포트 즉시 발송."""
    db = get_db()

    from trading import Portfolio, PaperEngine

    portfolio = Portfolio(PAPER_INITIAL_CAPITAL, trade_type="paper")
    latest = db.get_latest_portfolio("paper")
    if latest:
        portfolio.load_from_snapshot(latest)

    paper = PaperEngine(db, portfolio)
    report = paper.generate_report()

    print(report)

    try:
        from telegram_bot import send_message
        send_message(report)
        print("\n텔레그램 전송 완료")
    except Exception as e:
        print(f"\n텔레그램 전송 실패: {e}")


def cmd_backtest(args):
    """과거 데이터로 매매 시뮬레이션 → 후회 데이터 대량 생성.

    600일 일봉으로 매일 예측+매매를 재현하여:
    - 수천 건의 가상 매매 기록
    - 수백 건의 후회 분석 데이터
    - 이걸로 모델 재학습하면 실전 성능 대폭 향상
    """
    import numpy as np
    logger.info("=== 백테스트 시뮬레이션 시작 ===")
    db = get_db()

    from features import FeatureEngine
    from features.normalizer import FeatureNormalizer
    from models import LGBMModel, LSTMModel, EnsembleModel
    from trading import Portfolio
    from nn_config import (
        PAPER_COMMISSION_BUY, PAPER_COMMISSION_SELL,
        PAPER_TAX_RATE, PAPER_SLIPPAGE_BPS, CLASS_NAMES,
        PREDICTION_HORIZON, MAX_POSITIONS,
    )

    engine = FeatureEngine(db)
    lgbm = LGBMModel()
    lstm = LSTMModel()
    normalizer = FeatureNormalizer()

    lgbm.load(str(MODEL_DIR / "lgbm_latest.pkl"))
    try:
        lstm.load(str(MODEL_DIR / "lstm_latest.pt"))
    except Exception:
        logger.warning("LSTM 로드 실패 — LightGBM만 사용")
        lstm = None
    normalizer.load(str(MODEL_DIR / "normalizer.json"))
    ensemble = EnsembleModel(lgbm, lstm)
    feature_names = normalizer.get_feature_names()

    # 모든 종목의 거래일 목록 수집
    codes = list(ALL_STOCKS.keys())
    all_dates = db.execute_raw(
        "SELECT DISTINCT date FROM daily_prices ORDER BY date"
    )
    dates = [r['date'] for r in all_dates]

    # 최근 N일만 백테스트 (기본 400일, 앞 200일은 피처 계산에 필요)
    bt_days = min(args.days, len(dates) - 120)
    if bt_days < 30:
        print("백테스트할 데이터가 부족합니다 (최소 150일 필요)")
        return

    bt_dates = dates[-bt_days:]
    print(f"백테스트 기간: {bt_dates[0]} ~ {bt_dates[-1]} ({len(bt_dates)}일)")

    # 가상 포트폴리오
    portfolio = Portfolio(PAPER_INITIAL_CAPITAL, trade_type="backtest")
    bt_trades = []  # (date, code, side, price, quantity, predicted_class, confidence)
    bt_regrets = []

    total_buys = 0
    total_sells = 0

    for di, date in enumerate(bt_dates):
        # 해당일 가격 데이터
        day_prices = db.execute_raw(
            "SELECT stock_code, open, close, high, low, volume "
            "FROM daily_prices WHERE date=?", (date,)
        )
        if not day_prices:
            continue

        price_map = {}
        rate_map = {}
        vol_map = {}
        for dp in day_prices:
            code = dp['stock_code']
            price_map[code] = dp['close']
            if dp['open'] and dp['open'] > 0:
                rate_map[code] = (dp['close'] - dp['open']) / dp['open'] * 100
            vol_map[code] = dp.get('volume', 0) or 0

        # 현재가 갱신
        portfolio.update_prices(price_map)

        # 보유 종목 손절/익절 체크
        for code in list(portfolio.positions.keys()):
            pos = portfolio.positions[code]
            price = price_map.get(code, 0)
            if not price:
                continue
            pnl = (price - pos.avg_price) / pos.avg_price * 100

            if pnl < -5.0:
                # 긴급 손절 (안전장치)
                qty = pos.quantity
                commission = int(price * qty * PAPER_COMMISSION_SELL)
                tax = int(price * qty * PAPER_TAX_RATE)
                realized = portfolio.sell(code, qty, price, commission=commission, tax=tax)
                reason = f"긴급 손절 {pnl:.1f}%"
                bt_trades.append({
                    'date': date, 'stock_code': code, 'side': 'sell',
                    'price': price, 'quantity': qty, 'pnl': realized,
                    'reason': reason,
                })
                total_sells += 1

        # ML 예측 → 매수/매도
        predictions = []
        for code in codes:
            if code not in price_map:
                continue
            try:
                sector = SECTOR_MAP.get(code, "")
                feat = engine.compute_features_for_stock(code, date, sector)
                if not feat:
                    continue
                norm_feat = normalizer.transform(feat)
                x_flat = np.array([[norm_feat.get(k, 0) for k in feature_names]],
                                  dtype=np.float32)
                # 실제 시퀀스 (백테스트에서도)
                seq_data = engine.get_feature_sequence(code, date, feature_names, SEQUENCE_LENGTH)
                if seq_data:
                    norm_seq = []
                    for dv in [dict(zip(feature_names, day)) for day in seq_data]:
                        nd = normalizer.transform(dv)
                        norm_seq.append([nd.get(k, 0) for k in feature_names])
                    x_seq = np.array([norm_seq], dtype=np.float32)
                else:
                    x_seq = np.zeros((1, SEQUENCE_LENGTH, len(feature_names)), dtype=np.float32)
                    x_seq[0, -1, :] = x_flat[0]
                result = ensemble.predict_single(x_flat, x_seq)
                result['stock_code'] = code
                result['rate'] = rate_map.get(code, 0)
                result['volume'] = vol_map.get(code, 0)
                predictions.append(result)
            except Exception:
                pass

        # 매도 시그널 (보유 중 + ML 하락)
        for pred in predictions:
            code = pred['stock_code']
            if code in portfolio.positions and pred['predicted_class'] == 0 and pred['confidence'] > 0.5:
                pos = portfolio.positions[code]
                price = price_map[code]
                qty = pos.quantity
                commission = int(price * qty * PAPER_COMMISSION_SELL)
                tax = int(price * qty * PAPER_TAX_RATE)
                realized = portfolio.sell(code, qty, price, commission=commission, tax=tax)
                bt_trades.append({
                    'date': date, 'stock_code': code, 'side': 'sell',
                    'price': price, 'quantity': qty, 'pnl': realized,
                    'reason': f"ML {CLASS_NAMES[pred['predicted_class']]}",
                })
                total_sells += 1

        # 매수 시그널 (ML 상승 + 등락률 양수)
        buy_preds = [
            p for p in predictions
            if p['predicted_class'] == 2
            and p['confidence'] > 0.4
            and p['rate'] > 0.5
            and p['stock_code'] not in portfolio.positions
        ]
        buy_preds.sort(key=lambda x: -x['confidence'])

        for pred in buy_preds[:5]:  # 하루 최대 5종목 매수
            code = pred['stock_code']
            if portfolio.num_positions >= MAX_POSITIONS:
                break
            price = price_map[code]
            if price <= 0:
                continue

            # 포지션 사이징
            target = int(portfolio.total_value * 0.08 * pred['confidence'])
            qty = target // price
            if qty <= 0:
                continue

            commission = int(price * qty * PAPER_COMMISSION_BUY)
            slippage = int(price * qty * PAPER_SLIPPAGE_BPS / 10000)
            total_cost = price * qty + commission + slippage
            if total_cost > portfolio.cash:
                qty = int(portfolio.cash * 0.9) // price
                if qty <= 0:
                    continue

            name = ALL_STOCKS.get(code, code)
            success = portfolio.buy(code, name, qty, price,
                                    commission=commission, slippage=slippage)
            if success:
                bt_trades.append({
                    'date': date, 'stock_code': code, 'side': 'buy',
                    'price': price, 'quantity': qty,
                    'predicted_class': pred['predicted_class'],
                    'confidence': pred['confidence'],
                })
                total_buys += 1

        # 일일 수익률 기록
        portfolio.record_daily()

        # 진행 표시
        if (di + 1) % 20 == 0:
            print(f"  [{di+1}/{len(bt_dates)}] {date} | "
                  f"총평가 {portfolio.total_value:,}원 ({portfolio.cumulative_return:+.1f}%) | "
                  f"매수 {total_buys} 매도 {total_sells}")

    # === 후회 분석 (백테스트 매매에 대해) ===
    print(f"\n후회 분석 중... ({len(bt_trades)}건)")
    regret_count = 0
    for trade in bt_trades:
        code = trade['stock_code']
        trade_date = trade['date']

        # 매매 후 5일간 가격
        future = db.execute_raw(
            "SELECT close FROM daily_prices WHERE stock_code=? AND date>? ORDER BY date LIMIT ?",
            (code, trade_date, PREDICTION_HORIZON)
        )
        if len(future) < PREDICTION_HORIZON:
            continue

        trade_price = trade['price']
        future_price = future[-1]['close']
        ret_5d = (future_price - trade_price) / trade_price * 100

        side = trade['side']
        if side == 'buy' and ret_5d < -2:
            regret_score = min(1.0, abs(ret_5d) / 10)
            lesson = f"[BT] 매수 후 {ret_5d:.1f}% 하락"
        elif side == 'sell' and ret_5d > 2:
            regret_score = min(1.0, ret_5d / 10)
            lesson = f"[BT] 매도 후 {ret_5d:.1f}% 상승 — 조기 매도"
        else:
            regret_score = 0
            lesson = ""

        if regret_score > 0.2:
            db.insert_regret({
                'trade_id': 0,
                'stock_code': code,
                'trade_date': trade_date,
                'action_taken': f"{side} {trade.get('quantity',0)}주 @ {trade_price}",
                'optimal_action': '관망' if side == 'buy' else '보유 유지',
                'regret_score': regret_score,
                'return_if_optimal': -ret_5d if side == 'buy' else ret_5d,
                'return_actual': ret_5d if side == 'buy' else -ret_5d,
                'lesson': lesson,
            })
            regret_count += 1

    # 결과 출력
    from models.evaluator import Evaluator
    metrics = Evaluator.trading_metrics(portfolio.daily_returns)

    print(f"\n{'='*50}")
    print(f"백테스트 결과: {bt_dates[0]} ~ {bt_dates[-1]}")
    print(f"{'='*50}")
    print(f"  총 매수: {total_buys}건, 총 매도: {total_sells}건")
    print(f"  최종 평가: {portfolio.total_value:,}원")
    print(f"  누적 수익률: {portfolio.cumulative_return:+.2f}%")
    print(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  MDD: {metrics.get('max_drawdown', 0):.1f}%")
    print(f"  승률: {portfolio.win_rate:.1%}")
    print(f"  후회 데이터 생성: {regret_count}건")
    print(f"\n→ 'python3 nn_main.py train'으로 재학습하면 후회 데이터가 반영됩니다")


def cmd_forecast(args):
    """내일 급등 예측 기록 — 사후 검증용.

    사용법:
      python3 nn_main.py forecast 086520 +3.0 75
      python3 nn_main.py forecast 086520,009150,403870 +3.0,+2.0,+2.0 75,60,55
    """
    db = get_db()
    codes = args.codes.split(",")
    gaps = [float(g) for g in args.gap.split(",")]
    confs = [float(c) for c in args.confidence.split(",")]

    if len(gaps) == 1:
        gaps = gaps * len(codes)
    if len(confs) == 1:
        confs = confs * len(codes)

    today = datetime.now().strftime("%Y%m%d")

    for code, gap, conf in zip(codes, gaps, confs):
        name = ALL_STOCKS.get(code, code)
        pred_class = 2  # 상승
        db.insert_prediction({
            "stock_code": code,
            "date": today,
            "model_name": "human_forecast",
            "predicted_class": pred_class,
            "class_probabilities": [0, int(100 - conf), int(conf)],
            "confidence": conf / 100.0,
        })
        print(f"  예측 기록: {name}({code}) 내일 갭업 {gap:+.1f}%, 신뢰도 {conf:.0f}%")

    print(f"\n→ 내일 장 마감 후 'python3 nn_main.py verify'로 검증하세요")


def cmd_verify(args):
    """예측 검증 + 후회 학습 데이터 생성.

    human_forecast 예측의 actual_return을 채우고,
    틀린 예측은 regret_log에 기록 → 다음 학습에 반영.

    사용법:
      python3 nn_main.py verify              # 미검증 예측 전부
      python3 nn_main.py verify --date 20260219  # 특정 날짜
    """
    db = get_db()
    from nn_config import CLASS_NAMES

    target_date = args.date if hasattr(args, 'date') and args.date else None

    # 미검증 예측 (actual_return이 NULL인 것) 조회
    sql = """SELECT id, stock_code, date, model_name, predicted_class, confidence
             FROM predictions
             WHERE actual_return IS NULL"""
    params = ()
    if target_date:
        sql += " AND date = ?"
        params = (target_date,)
    sql += " ORDER BY date ASC"

    unverified = db.execute_raw(sql, params)
    if not unverified:
        print("검증할 예측이 없습니다.")
        return

    print(f"미검증 예측 {len(unverified)}건 처리 중...\n")

    verified = 0
    regrets_added = 0
    results = []

    for pred in unverified:
        code = pred['stock_code']
        pred_date = pred['date']

        # 예측일의 종가
        base = db.execute_raw(
            "SELECT close FROM daily_prices WHERE stock_code=? AND date=?",
            (code, pred_date)
        )
        if not base or not base[0]['close']:
            continue

        # 다음 거래일의 시가, 고가, 종가
        next_days = db.execute_raw(
            "SELECT date, open, high, low, close FROM daily_prices "
            "WHERE stock_code=? AND date>? ORDER BY date LIMIT 5",
            (code, pred_date)
        )
        if not next_days:
            continue  # 아직 다음날 데이터 없음

        base_close = base[0]['close']
        nd = next_days[0]  # 다음 거래일

        gap_pct = (nd['open'] - base_close) / base_close * 100 if nd['open'] else 0
        high_pct = (nd['high'] - base_close) / base_close * 100 if nd['high'] else 0
        close_pct = (nd['close'] - base_close) / base_close * 100 if nd['close'] else 0

        # 5일 수익률 (있으면)
        if len(next_days) >= 5:
            ret_5d = (next_days[4]['close'] - base_close) / base_close * 100
        else:
            ret_5d = close_pct  # 폴백: 1일 수익률

        # actual_class 계산 (3-class)
        if ret_5d < -1.5:
            actual_class = 0  # 하락
        elif ret_5d > 1.5:
            actual_class = 2  # 상승
        else:
            actual_class = 1  # 보합

        # 예측 업데이트
        db.update_prediction_actual(pred['id'], actual_class, ret_5d)

        name = ALL_STOCKS.get(code, code)
        pred_class = pred['predicted_class']
        hit = "O" if pred_class == actual_class else ("~" if abs(pred_class - actual_class) <= 1 else "X")

        results.append({
            'code': code, 'name': name, 'pred_date': pred_date,
            'pred_class': pred_class, 'actual_class': actual_class,
            'gap': gap_pct, 'high': high_pct, 'close': close_pct,
            'ret_5d': ret_5d, 'hit': hit, 'model': pred['model_name'],
        })
        verified += 1

        # 후회 분석: 상승(2) 예측했는데 실제 하락(-1% 이하)이면 후회 기록
        if pred_class == 2 and ret_5d < -1:
            regret_score = min(1.0, abs(ret_5d) / 10)
            db.insert_regret({
                'trade_id': 0,
                'stock_code': code,
                'trade_date': pred_date,
                'action_taken': f"예측 {CLASS_NAMES[pred_class]} (신뢰도 {pred['confidence']:.1%})",
                'optimal_action': '관망' if ret_5d < -3 else '소량매수',
                'regret_score': regret_score,
                'return_if_optimal': 0,
                'return_actual': ret_5d,
                'lesson': f"[검증] {name} {CLASS_NAMES[pred_class]} 예측 → 실제 {ret_5d:+.1f}% ({CLASS_NAMES[actual_class]})",
            })
            regrets_added += 1

        # 반대: 하락(0) 예측했는데 실제 급등(+3% 이상)
        elif pred_class == 0 and ret_5d > 3:
            regret_score = min(1.0, ret_5d / 10)
            db.insert_regret({
                'trade_id': 0,
                'stock_code': code,
                'trade_date': pred_date,
                'action_taken': f"예측 {CLASS_NAMES[pred_class]} (신뢰도 {pred['confidence']:.1%})",
                'optimal_action': '매수',
                'regret_score': regret_score,
                'return_if_optimal': ret_5d,
                'return_actual': 0,
                'lesson': f"[검증] {name} {CLASS_NAMES[pred_class]} 예측 → 실제 {ret_5d:+.1f}% ({CLASS_NAMES[actual_class]}) — 놓침",
            })
            regrets_added += 1

    # human_forecast만 따로 표시
    human = [r for r in results if r['model'] == 'human_forecast']
    ml = [r for r in results if r['model'] == 'ensemble']

    if human:
        print(f"{'='*60}")
        print(f" 수동 예측 검증 결과")
        print(f"{'='*60}")
        for r in human:
            print(f"  {r['hit']} {r['name']}({r['code']}) | "
                  f"예측: {CLASS_NAMES[r['pred_class']]} → 실제: {CLASS_NAMES[r['actual_class']]} | "
                  f"갭 {r['gap']:+.1f}% 고점 {r['high']:+.1f}% 종가 {r['close']:+.1f}%")
        hits = sum(1 for r in human if r['hit'] == 'O')
        near = sum(1 for r in human if r['hit'] == '~')
        print(f"\n  적중: {hits}/{len(human)} ({hits/len(human)*100:.0f}%), "
              f"근접: {near}/{len(human)}")

    if ml:
        ml_hits = sum(1 for r in ml if r['hit'] in ('O', '~'))
        print(f"\n  ML 앙상블: {len(ml)}건 검증, 적중+근접 {ml_hits}/{len(ml)} "
              f"({ml_hits/len(ml)*100:.0f}%)")

    print(f"\n총 검증: {verified}건, 후회 데이터 생성: {regrets_added}건")
    if regrets_added > 0:
        print(f"→ 'python3 nn_main.py train'으로 재학습하면 후회 데이터가 반영됩니다")


def cmd_dashboard(args):
    """웹 대시보드 시작."""
    from dashboard.app import run_dashboard
    run_dashboard()


def main():
    parser = argparse.ArgumentParser(description="신경망 자동매매 시스템")
    sub = parser.add_subparsers(dest="command")

    # backfill
    p_bf = sub.add_parser("backfill", aliases=["--backfill"], help="과거 데이터 백필")
    p_bf.add_argument("--days", type=int, default=BACKFILL_DAYS)

    # features
    sub.add_parser("features", aliases=["--features"], help="피처 계산")

    # train
    p_train = sub.add_parser("train", aliases=["--train"], help="모델 학습")
    p_train.add_argument("--walk-forward", action="store_true", help="Walk-forward CV")

    # paper
    sub.add_parser("paper", aliases=["--paper"], help="모의투자 시작")

    # predict (즉시 예측)
    sub.add_parser("predict", aliases=["--predict"], help="즉시 예측+매매")

    # status
    sub.add_parser("status", aliases=["--status"], help="상태 조회")

    # report
    sub.add_parser("report", aliases=["--report"], help="텔레그램 리포트")

    # backtest
    p_bt = sub.add_parser("backtest", aliases=["--backtest"], help="과거 데이터 백테스트")
    p_bt.add_argument("--days", type=int, default=400, help="백테스트 일수")

    # forecast (내일 급등 예측 기록)
    p_fc = sub.add_parser("forecast", help="내일 급등 예측 기록 (검증용)")
    p_fc.add_argument("codes", help="종목코드 (콤마 구분)")
    p_fc.add_argument("gap", help="예상 갭업% (콤마 구분)")
    p_fc.add_argument("confidence", help="신뢰도% (콤마 구분)")

    # verify (예측 검증 + 후회학습)
    p_vf = sub.add_parser("verify", help="예측 검증 + 후회 데이터 생성")
    p_vf.add_argument("--date", default=None, help="검증할 날짜 (YYYYMMDD)")

    # dashboard
    sub.add_parser("dashboard", aliases=["--dashboard"], help="대시보드")

    args = parser.parse_args()

    # --backfill 스타일 지원
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        cmd = sys.argv[1].lstrip("-")
        sys.argv[1] = cmd

        # re-parse
        args = parser.parse_args()

    cmd_map = {
        "backfill": cmd_backfill, "--backfill": cmd_backfill,
        "features": cmd_features, "--features": cmd_features,
        "train": cmd_train, "--train": cmd_train,
        "paper": cmd_paper, "--paper": cmd_paper,
        "predict": cmd_predict, "--predict": cmd_predict,
        "status": cmd_status, "--status": cmd_status,
        "report": cmd_report, "--report": cmd_report,
        "backtest": cmd_backtest, "--backtest": cmd_backtest,
        "forecast": cmd_forecast,
        "verify": cmd_verify,
        "dashboard": cmd_dashboard, "--dashboard": cmd_dashboard,
    }

    if args.command in cmd_map:
        cmd_map[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
