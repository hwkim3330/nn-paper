"""
TimesFM-3 선택 기능 테스트 (timesfm 미설치 환경에서도 실행 가능)

  python -m unittest tests.test_timesfm_optional -v
  RUN_TIMESFM=1 python -m unittest tests.test_timesfm_optional -v   # 실제 모델 추론까지 (가중치 ~1.3GB)
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("NN_PAPER_LIVE_TRADING", None)
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


class TestWithoutTimesFM(unittest.TestCase):
    """timesfm3 import를 막은 상태에서 기존 파이프라인 모듈이 그대로 import 되는지."""

    BLOCK = "import sys; sys.modules['timesfm3'] = None; sys.modules['timesfm3.mlx'] = None; " \
            "sys.modules['timesfm3.torch'] = None\n"

    def test_pipeline_imports(self):
        r = _run(self.BLOCK + (
            "import features, models, trading, db\n"
            "from models import LGBMModel, Trainer, Evaluator\n"
            "from trading.live_engine import LiveEngine\n"
            "import models.timesfm_forecaster as t\n"
            "assert not t.is_available()\n"
            "print('ok')\n"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ok", r.stdout)

    def test_forecaster_raises_helpful_import_error(self):
        r = _run(self.BLOCK + (
            "from models.timesfm_forecaster import TimesFMForecaster\n"
            "try:\n"
            "    TimesFMForecaster()\n"
            "except ImportError as e:\n"
            "    assert 'timesfm[mlx]' in str(e), e\n"
            "    print('ok')\n"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ok", r.stdout)


class TestLiveGuard(unittest.TestCase):

    def test_live_process_refuses_timesfm(self):
        r = _run(
            "from trading.live_engine import LiveEngine\n"
            "from models.timesfm_forecaster import TimesFMForecaster, TimesFMLicenseError\n"
            "class P: daily_returns = []\n"
            "LiveEngine(db=None, portfolio=P())\n"
            "try:\n"
            "    TimesFMForecaster()\n"
            "    print('NOT BLOCKED')\n"
            "except TimesFMLicenseError:\n"
            "    print('blocked')\n"
            "try:\n"
            "    LiveEngine.check_model_features(['rsi_14', 'tfm_p_up_1d'])\n"
            "    print('NOT BLOCKED')\n"
            "except TimesFMLicenseError:\n"
            "    print('blocked')\n"
            "LiveEngine.check_model_features(['rsi_14'])\n")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split().count("blocked"), 2, r.stdout)
        self.assertNotIn("NOT BLOCKED", r.stdout)

    def test_env_flag_blocks(self):
        env = dict(os.environ, NN_PAPER_LIVE_TRADING="1")
        r = subprocess.run([sys.executable, "-c",
                            "from models.timesfm_forecaster import assert_research_use\n"
                            "assert_research_use()"],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("TimesFMLicenseError", r.stderr)


class TestQuantileMath(unittest.TestCase):

    def setUp(self):
        from models.timesfm_forecaster import QUANTILE_LEVELS
        from statistics import NormalDist
        self.nd = NormalDist(0.001, 0.02)
        self.q = np.array([self.nd.inv_cdf(t) for t in QUANTILE_LEVELS])

    def test_prob_above_matches_normal(self):
        from models.timesfm_forecaster import prob_above
        for x in (-0.05, -0.01, 0.0, 0.02, 0.05):
            self.assertAlmostEqual(float(prob_above(self.q, x)), 1 - self.nd.cdf(x), delta=0.01)

    def test_barrier_probs_simplex(self):
        from models.timesfm_forecaster import barrier_probs
        pq = np.stack([self.q * np.sqrt(k) for k in range(1, 16)])[None].repeat(3, 0)
        p = barrier_probs(pq, np.array([0.01, 0.05, 0.3]))
        np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        self.assertTrue((p >= 0).all())
        self.assertGreater(p[2, 1], p[0, 1])   # barrier 넓을수록 보합 확률 ↑

    def test_signal_format(self):
        from models.timesfm_forecaster import TimesFMForecaster
        pq = np.stack([self.q * np.sqrt(k) for k in range(1, 16)])[None]
        sig = TimesFMForecaster.signal_from_quantiles(pq, np.array([0.05]), ["005930"])[0]
        for k in ("stock_code", "predicted_class", "confidence", "buy_prob", "sell_prob", "signal"):
            self.assertIn(k, sig)
        self.assertIn(sig["predicted_class"], (0, 1, 2))

    def test_crps_zero_at_degenerate(self):
        from models.timesfm_forecaster import crps_from_quantiles
        self.assertAlmostEqual(float(crps_from_quantiles(np.zeros(9), 0.0)), 0.0)


@unittest.skipUnless(os.environ.get("RUN_TIMESFM") == "1", "RUN_TIMESFM=1 일 때만 (모델 다운로드)")
class TestTimesFMInference(unittest.TestCase):

    def test_features_for_stock(self):
        from models.timesfm_forecaster import TimesFMForecaster
        rng = np.random.default_rng(0)
        c = 50000 * np.exp(np.cumsum(rng.normal(0, 0.02, 600)))
        prices = [{"date": str(i), "open": x, "high": x * 1.01, "low": x * 0.99, "close": x,
                   "volume": 1e6} for i, x in enumerate(c)]
        fc = TimesFMForecaster()
        f = fc.features_for_stock(prices)
        self.assertTrue(all(k.startswith("tfm_") for k in f))
        for k in ("tfm_p_up_1d", "tfm_exp_ret_5d", "tfm_spread_5d", "tfm_rv_5d", "tfm_tb_p_up"):
            self.assertIn(k, f)
            self.assertTrue(np.isfinite(f[k]))
        sig = fc.signal(prices, "TEST")
        self.assertIn(sig["predicted_class"], (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
