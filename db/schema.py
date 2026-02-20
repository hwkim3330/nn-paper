"""
SQLite 테이블 스키마 정의
- 12개 테이블: 가격, 호가, 수급, 프로그램, 업종, 시장, 피처, 예측, 매매, 포트폴리오, 학습로그, 후회로그
"""
import sqlite3
import logging

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
-- 일봉 데이터
CREATE TABLE IF NOT EXISTS daily_prices (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    open INTEGER,
    high INTEGER,
    low INTEGER,
    close INTEGER,
    volume INTEGER,
    change_rate REAL,
    market_cap INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_prices(date);

-- 분봉 데이터
CREATE TABLE IF NOT EXISTS minute_prices (
    stock_code TEXT NOT NULL,
    datetime TEXT NOT NULL,
    interval_min INTEGER NOT NULL,
    open INTEGER,
    high INTEGER,
    low INTEGER,
    close INTEGER,
    volume INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, datetime, interval_min)
);

-- 호가 스냅샷
CREATE TABLE IF NOT EXISTS orderbook_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    datetime TEXT NOT NULL,
    total_ask_volume INTEGER,
    total_bid_volume INTEGER,
    ask_bid_ratio REAL,
    spread_pct REAL,
    ask_prices TEXT,     -- JSON: [price1, price2, ...]
    ask_volumes TEXT,    -- JSON: [vol1, vol2, ...]
    bid_prices TEXT,     -- JSON
    bid_volumes TEXT,    -- JSON
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_ob_code_dt ON orderbook_snapshots(stock_code, datetime);

-- 투자자별 매매동향 (외국인/기관)
CREATE TABLE IF NOT EXISTS investor_flow (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    foreign_buy INTEGER DEFAULT 0,
    foreign_sell INTEGER DEFAULT 0,
    foreign_net INTEGER DEFAULT 0,
    institution_buy INTEGER DEFAULT 0,
    institution_sell INTEGER DEFAULT 0,
    institution_net INTEGER DEFAULT 0,
    individual_net INTEGER DEFAULT 0,
    foreign_hold_ratio REAL,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, date)
);

-- 프로그램 매매
CREATE TABLE IF NOT EXISTS program_trading (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    program_buy INTEGER DEFAULT 0,
    program_sell INTEGER DEFAULT 0,
    program_net INTEGER DEFAULT 0,
    arbitrage_buy INTEGER DEFAULT 0,
    arbitrage_sell INTEGER DEFAULT 0,
    non_arbitrage_buy INTEGER DEFAULT 0,
    non_arbitrage_sell INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, date)
);

-- 업종/테마 데이터
CREATE TABLE IF NOT EXISTS sector_data (
    sector_code TEXT NOT NULL,
    date TEXT NOT NULL,
    sector_name TEXT,
    close REAL,
    change_rate REAL,
    volume INTEGER,
    market_cap INTEGER,
    advance_count INTEGER,
    decline_count INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (sector_code, date)
);

-- 시장 데이터 (US, 환율, VIX 등)
CREATE TABLE IF NOT EXISTS market_data (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    price REAL,
    change REAL,
    change_pct REAL,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (date, symbol)
);

-- 계산된 피처 벡터
CREATE TABLE IF NOT EXISTS features (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    feature_vector TEXT NOT NULL,  -- JSON: {feature_name: value, ...}
    feature_version INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, date)
);

-- 모델 예측
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    model_name TEXT NOT NULL,       -- 'lgbm', 'lstm', 'ensemble'
    predicted_class INTEGER,        -- 0~4
    class_probabilities TEXT,       -- JSON: [p0, p1, p2, p3, p4]
    confidence REAL,
    actual_class INTEGER,           -- 사후 실제 클래스 (후회 학습용)
    actual_return REAL,             -- 사후 실제 수익률
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_pred_code_dt ON predictions(stock_code, date);

-- 매매 내역
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_type TEXT NOT NULL,       -- 'paper' or 'live'
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    side TEXT NOT NULL,             -- 'buy' or 'sell'
    quantity INTEGER NOT NULL,
    price INTEGER NOT NULL,
    amount INTEGER NOT NULL,        -- 총 금액
    commission INTEGER DEFAULT 0,
    tax INTEGER DEFAULT 0,
    slippage INTEGER DEFAULT 0,
    predicted_class INTEGER,
    confidence REAL,
    reason TEXT,                    -- 매매 사유
    -- 후회 분석 (사후 기록)
    regret_score REAL,              -- 후회 점수 (양수=좋은 결정, 음수=나쁜 결정)
    regret_analysis TEXT,           -- 후회 분석 내용
    actual_return_5d REAL,          -- 매매 후 5일 수익률
    optimal_action TEXT,            -- 사후 최적 행동
    executed_at TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_trades_dt ON trades(executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_code ON trades(stock_code);

-- 포트폴리오 상태 (일별 스냅샷)
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    trade_type TEXT NOT NULL,       -- 'paper' or 'live'
    total_value INTEGER,            -- 총 평가금
    cash INTEGER,                   -- 현금
    stock_value INTEGER,            -- 주식 평가액
    daily_return REAL,              -- 일간 수익률
    cumulative_return REAL,         -- 누적 수익률
    positions TEXT,                 -- JSON: [{code, name, qty, avg_price, current_price, pnl}, ...]
    num_positions INTEGER,
    max_drawdown REAL,
    sharpe_ratio REAL,
    win_rate REAL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_portfolio_dt ON portfolio_snapshots(date, trade_type);

-- 학습 로그
CREATE TABLE IF NOT EXISTS training_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    train_start_date TEXT,
    train_end_date TEXT,
    valid_start_date TEXT,
    valid_end_date TEXT,
    metrics TEXT,                   -- JSON: {accuracy, sharpe, drawdown, ...}
    model_path TEXT,
    feature_importance TEXT,        -- JSON: {feature: importance, ...}
    training_time_sec REAL,
    created_at TEXT DEFAULT (datetime('now','localtime'))
);

-- 후회 로그 (핵심: 매매 후 "이렇게 했어야" 기록)
CREATE TABLE IF NOT EXISTS regret_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER REFERENCES trades(id),
    stock_code TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    action_taken TEXT NOT NULL,       -- 실제 취한 행동
    optimal_action TEXT NOT NULL,     -- 사후 최적 행동
    regret_score REAL NOT NULL,       -- 후회 크기 (0~1)
    return_if_optimal REAL,           -- 최적 행동 시 수익률
    return_actual REAL,               -- 실제 수익률
    lesson TEXT,                       -- 학습 교훈 (텍스트)
    applied_to_training INTEGER DEFAULT 0,  -- 학습에 반영했는지
    created_at TEXT DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_regret_dt ON regret_log(trade_date);

-- 순위 데이터 (거래량/등락률 등)
CREATE TABLE IF NOT EXISTS ranking_data (
    date TEXT NOT NULL,
    ranking_type TEXT NOT NULL,     -- 'volume', 'change_up', 'change_down', 'market_cap'
    rank_num INTEGER NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    price INTEGER,
    change_rate REAL,
    volume INTEGER,
    amount INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (date, ranking_type, rank_num)
);

-- NXT 야간거래 가격 히스토리
CREATE TABLE IF NOT EXISTS nxt_prices (
    stock_code TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,           -- HH:MM
    nxt_price INTEGER,
    krx_close INTEGER,            -- 당일 장중 종가
    nxt_vs_close_pct REAL,        -- NXT가 대비 종가 변화율
    spread_pct REAL,              -- KRX↔NXT 스프레드
    volume INTEGER,
    created_at TEXT DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (stock_code, date, time)
);
CREATE INDEX IF NOT EXISTS idx_nxt_date ON nxt_prices(date);
""";


def create_tables(db_path: str):
    """모든 테이블 생성."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("DB 테이블 생성 완료: %s", db_path)
    finally:
        conn.close()
