#!/usr/bin/env python3
"""
stock-bot DB에서 가격/수급/시장 데이터를 nn-paper DB로 동기화.
피처/모델/매매 데이터는 동기화하지 않음 (v2 자체 생성).
"""
import sqlite3
import sys
import os
from pathlib import Path

# 경로
STOCKBOT_DB = Path("/home/kim/stock-bot/data/stock_data.db")
NNPAPER_DB = Path(__file__).parent.parent / "data" / "stock_data.db"

# 동기화할 테이블 (읽기 전용 — 가격/수급만)
SYNC_TABLES = [
    "daily_prices",
    "minute_prices",
    "investor_flow",
    "program_trading",
    "sector_data",
    "market_data",
    "ranking_data",
    "nxt_prices",
]


def sync():
    if not STOCKBOT_DB.exists():
        print(f"ERROR: stock-bot DB 없음: {STOCKBOT_DB}")
        sys.exit(1)

    NNPAPER_DB.parent.mkdir(parents=True, exist_ok=True)

    src = sqlite3.connect(str(STOCKBOT_DB))
    src.row_factory = sqlite3.Row

    # nn-paper DB 초기화 (스키마 생성)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.schema import create_tables
    create_tables(str(NNPAPER_DB))
    dst = sqlite3.connect(str(NNPAPER_DB))

    total = 0
    for table in SYNC_TABLES:
        try:
            # 소스 테이블 존재 확인
            src_count = src.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            dst_count = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

            if src_count == dst_count:
                print(f"  {table}: {src_count}행 (이미 동기화)")
                continue

            # 컬럼 목록
            cols_info = src.execute(f"PRAGMA table_info({table})").fetchall()
            cols = [c[1] for c in cols_info]
            cols_str = ", ".join(cols)
            placeholders = ", ".join(["?"] * len(cols))

            # 기존 데이터 삭제 후 전체 복사 (간단하게)
            dst.execute(f"DELETE FROM {table}")

            batch_size = 5000
            offset = 0
            inserted = 0
            while True:
                rows = src.execute(
                    f"SELECT {cols_str} FROM {table} LIMIT {batch_size} OFFSET {offset}"
                ).fetchall()
                if not rows:
                    break
                dst.executemany(
                    f"INSERT OR REPLACE INTO {table} ({cols_str}) VALUES ({placeholders})",
                    [tuple(r) for r in rows],
                )
                inserted += len(rows)
                offset += batch_size

            dst.commit()
            print(f"  {table}: {inserted}행 동기화 완료 (src={src_count})")
            total += inserted

        except Exception as e:
            print(f"  {table}: ERROR — {e}")

    src.close()
    dst.close()
    print(f"\n총 {total}행 동기화 완료 → {NNPAPER_DB}")


if __name__ == "__main__":
    print(f"=== stock-bot → nn-paper 데이터 동기화 ===")
    print(f"  소스: {STOCKBOT_DB}")
    print(f"  대상: {NNPAPER_DB}")
    print()
    sync()
