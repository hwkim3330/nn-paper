#!/bin/bash
# nn-paper v2: 3-class Triple Barrier + ML 기반 청산
# 구 stock-bot NN과 병행 운영 (A/B 테스트)

cd "$(dirname "$0")"

case "${1:-help}" in
  sync)
    echo "=== stock-bot DB에서 가격/수급 데이터 동기화 ==="
    python3 scripts/sync_from_stockbot.py
    ;;
  features)
    echo "=== 피처 재계산 (v2: 93개, triple barrier) ==="
    python3 nn_main.py features
    ;;
  train)
    echo "=== 모델 학습 (3-class) ==="
    python3 nn_main.py train
    ;;
  paper)
    echo "=== 모의투자 v2 시작 ==="
    python3 nn_main.py paper
    ;;
  backtest)
    echo "=== 백테스트 ==="
    shift
    python3 nn_main.py backtest "$@"
    ;;
  status)
    python3 nn_main.py status
    ;;
  report)
    python3 nn_main.py report
    ;;
  dashboard)
    echo "=== 대시보드 (port 8502) ==="
    python3 nn_main.py dashboard
    ;;
  predict)
    shift
    python3 nn_main.py predict "$@"
    ;;
  verify)
    shift
    python3 nn_main.py verify "$@"
    ;;
  setup)
    echo "=== 초기 세팅: sync → features → train ==="
    python3 scripts/sync_from_stockbot.py
    python3 nn_main.py features
    python3 nn_main.py train
    echo "=== 완료! ./run.sh paper 로 모의투자 시작 ==="
    ;;
  *)
    echo "nn-paper v2 — 3-class Triple Barrier + ML 청산"
    echo ""
    echo "사용법: ./run.sh <command>"
    echo ""
    echo "  setup     초기 세팅 (sync → features → train)"
    echo "  sync      stock-bot DB에서 가격 데이터 동기화"
    echo "  features  피처 재계산"
    echo "  train     모델 학습"
    echo "  paper     모의투자 시작"
    echo "  backtest  백테스트 (--days N)"
    echo "  status    현재 상태"
    echo "  report    텔레그램 리포트"
    echo "  dashboard 웹 대시보드 (port 8502)"
    echo "  predict   내일 예측"
    echo "  verify    예측 검증"
    ;;
esac
