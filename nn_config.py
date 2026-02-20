"""
신경망 자동매매 시스템 설정
- 후회 기반 학습: 매매 결과를 분석하여 "이렇게 했어야" 를 모델에 반영
- 모의투자로 무한 반복 후 기준 충족 시 실전 전환
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "stock_data.db"
MODEL_DIR = DATA_DIR / "models"

# 디렉토리 자동 생성
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 대상 종목 (KOSPI200 + KOSDAQ150 + 관심종목)
# ============================================================
# config.py에서 가져옴 — 중복 정의 방지
from config import WATCHLIST, KOSPI200_TOP, KOSDAQ150_TOP

# 전체 대상 종목 (합집합)
ALL_STOCKS = {}
ALL_STOCKS.update(KOSPI200_TOP)
ALL_STOCKS.update(KOSDAQ150_TOP)
ALL_STOCKS.update(WATCHLIST)

# ============================================================
# API Rate Limiting
# ============================================================
API_CALLS_PER_SEC = 5
API_CALL_INTERVAL = 1.0 / API_CALLS_PER_SEC  # 0.2초

# ============================================================
# 데이터 수집 설정
# ============================================================
BACKFILL_DAYS = 600          # 과거 데이터 백필 일수
MINUTE_INTERVALS = [5, 15]   # 분봉 간격 (분)
ORDERBOOK_SNAPSHOT_INTERVAL = 60  # 호가 스냅샷 간격 (초, 장중)

# 수집 스케줄 (KST)
COLLECT_SCHEDULE = {
    "daily_price": "16:30",      # 장후 일봉 수집
    "minute_price": "16:00",     # 장후 분봉 수집
    "orderbook": "market_hours", # 장중 매분
    "investor_flow": "16:30",    # 장후 외국인/기관
    "program_trading": "16:30",  # 장후 프로그램
    "sector": "16:30",           # 장후 업종
    "market": "08:00",           # 장전 US/환율
    "ranking": "15:40",          # 장마감 직후 순위
}

# ============================================================
# Feature Engineering 설정
# ============================================================
# 기술적 지표 파라미터
MA_PERIODS = [5, 10, 20, 60, 120]
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14
STOCH_K = 14
STOCH_D = 3
ADX_PERIOD = 14
CCI_PERIOD = 20
WILLIAMS_R_PERIOD = 14
OBV_MA_PERIOD = 20

# 시퀀스 길이 (LSTM 입력)
SEQUENCE_LENGTH = 60  # 60 거래일 (한국시장 연구: 60~100일 최적)

# 전체 피처 수 (자동 계산됨, 초기 추정)
EXPECTED_FEATURES = 93  # 호가 13개 제거, 새 피처 6개 추가 (101-13+6-1=93)

# ============================================================
# 모델 설정
# ============================================================
# LightGBM
LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,               # 3-class: 하락/보합/상승
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "n_estimators": 500,          # 트리 수 축소 (과소적합 방지)
    "max_depth": 6,
    "learning_rate": 0.03,        # 빠르게 → 과소적합 방지
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,             # L1 규제 완화
    "reg_lambda": 0.1,            # L2 규제 완화
    "is_unbalance": True,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 10,
}

# LSTM
LSTM_PARAMS = {
    "input_size": EXPECTED_FEATURES,
    "hidden_size": 64,            # 작게 → 한국시장에선 오히려 나음
    "num_layers": 2,
    "dropout": 0.2,               # 드롭아웃 완화 → 과소적합 방지
    "attention": True,
    "num_classes": 3,             # 3-class: 하락/보합/상승
    "learning_rate": 0.001,       # LR 높이기
    "batch_size": 128,            # 배치 키우기
    "epochs": 100,
    "patience": 30,               # 인내심 키우기
    "device": "cuda",
}

# 앙상블 가중치 (초기값 — 동적 조정됨)
ENSEMBLE_WEIGHTS = {
    "lgbm": 0.6,
    "lstm": 0.4,
}
ENSEMBLE_DYNAMIC = True  # 최근 30일 정확도 기반 자동 조정

# 3-class Triple Barrier 분류
# class 0: 하락 (stop barrier hit)
# class 1: 보합 (timeout — 15일 내 barrier 안 닿음)
# class 2: 상승 (profit barrier hit)
NUM_CLASSES = 3
CLASS_NAMES = ["하락", "보합", "상승"]
PREDICTION_HORIZON = 5  # 예측 기본 horizon

# Triple Barrier 설정
TRIPLE_BARRIER = {
    "atr_period": 14,
    "atr_multiplier": 1.5,    # ±1.5×ATR barrier
    "max_holding_days": 15,   # timeout
}

# 구 5-class 설정 (호환용)
RETURN_BINS = [-float('inf'), -3.0, -1.0, 1.0, 3.0, float('inf')]

# Walk-forward 학습
TRAIN_WINDOW = 400       # 학습 윈도우 (거래일)
VALID_WINDOW = 60        # 검증 윈도우
TEST_WINDOW = 20         # 테스트 윈도우
RETRAIN_INTERVAL = 5     # 재학습 주기 (거래일) — 주간

# 후회 기반 학습 (Regret-Based Learning)
REGRET_LOOKBACK = 5      # 매매 후 N일 뒤 결과 평가
REGRET_WEIGHT_BOOST = 5.0  # 후회 샘플 가중치 강화 (2.0→5.0)

# 시간 감쇠 가중치 (최신 데이터에 높은 가중치)
TIME_DECAY_HALFLIFE = 120  # 120일 반감기

# ============================================================
# 모의투자 설정
# ============================================================
PAPER_INITIAL_CAPITAL = 100_000_000  # 1억원
PAPER_COMMISSION_BUY = 0.00015       # 매수 수수료 0.015%
PAPER_COMMISSION_SELL = 0.00015      # 매도 수수료 0.015%
PAPER_TAX_RATE = 0.0018              # 거래세 0.18%
PAPER_SLIPPAGE_BPS = 5               # 슬리피지 5bp (0.05%)

# 포지션 제한
MAX_POSITIONS = 15         # 최대 보유 종목 수 (더 분산)
MAX_POSITION_PCT = 0.12    # 종목당 최대 비중 12%
MAX_SECTOR_PCT = 0.30      # 섹터당 최대 비중 30%

# 리스크 관리 — ML 기반 청산
# 손절/익절은 ML 신호로 판단 (고정 % X)
# 극단적 손실(-5%)만 절대 안전장치
TRAILING_STOP_PCT = -5.0   # 트레일링 스톱 -5% (고점 대비, 수익 보호)

# Kelly Criterion 파라미터
KELLY_FRACTION = 0.15      # 반 Kelly → 안정적 복리
MIN_POSITION_SIZE = 0.05   # 최소 포지션 5%

# 일일 리스크 제한
DAILY_LOSS_LIMIT = -1.5    # 일일 최대 손실 -1.5% (넘으면 당일 매매 중단)
CONSECUTIVE_LOSS_DAYS = 3  # 연속 N일 손실 시 포지션 축소
CONSECUTIVE_LOSS_SCALE = 0.5  # 포지션 사이즈 50% 축소

# ============================================================
# 실전 전환 기준 (모두 충족 시)
# ============================================================
LIVE_CRITERIA = {
    "min_days": 60,            # 최소 60일 모의투자
    "min_cumulative_return": 0, # 누적 수익 양수
    "min_sharpe": 0.5,         # Sharpe ratio > 0.5
    "max_drawdown": -15.0,     # Max drawdown < 15%
    "min_win_rate": 0.45,      # 승률 > 45%
}

# 실전 단계적 자금 투입
LIVE_CAPITAL_STAGES = [
    (0.01, "1단계: 1% 투입"),
    (0.05, "2단계: 5% 투입"),
    (0.20, "3단계: 20% 투입"),
    (0.50, "4단계: 50% 투입"),
    (1.00, "5단계: 전액 투입"),
]

# ============================================================
# 섹터 분류
# ============================================================
SECTOR_MAP = {
    # 반도체/AI
    "005930": "반도체", "000660": "반도체", "009150": "반도체",
    # 금융
    "316140": "금융", "055550": "금융", "105560": "금융", "086790": "금융",
    "032830": "금융", "000810": "금융",
    # 방산/조선
    "012450": "방산", "064350": "방산", "329180": "조선", "009540": "조선",
    # 자동차
    "005380": "자동차", "000270": "자동차", "012330": "자동차",
    # IT/플랫폼
    "066570": "IT", "035420": "IT", "035720": "IT", "018260": "IT",
    "259960": "IT", "036570": "IT", "064400": "IT",
    # 배터리/에너지
    "006400": "배터리", "373220": "배터리", "015760": "에너지",
    "051910": "화학", "003670": "배터리", "096770": "에너지",
    # 바이오
    "207940": "바이오", "196170": "바이오", "068270": "바이오", "145020": "바이오",
    # 기타
    "005490": "철강", "010130": "비철", "033780": "소비재",
    "017670": "통신", "030200": "통신", "003490": "운송", "011200": "운송",
    "034020": "중공업", "028260": "지주", "034730": "지주", "003550": "지주",
    "010950": "정유", "352820": "엔터", "041510": "엔터",
    "247540": "배터리", "086520": "배터리",
    "263750": "IT", "293490": "IT", "067160": "IT",
    "403870": "반도체", "257720": "소비재",
}

# ============================================================
# 텔레그램 리포트 설정
# ============================================================
REPORT_SCHEDULE = {
    "daily_summary": "20:00",    # 일일 모의투자 성적표
    "weekly_summary": "SAT_10",  # 주간 요약 (토요일 10시)
}

# ============================================================
# 대시보드 설정
# ============================================================
DASHBOARD_PORT = 8502  # v2 (stock-bot은 8501)
DASHBOARD_HOST = "0.0.0.0"

# ============================================================
# 로깅
# ============================================================
NN_LOG_FILE = BASE_DIR / "nn_trading.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
