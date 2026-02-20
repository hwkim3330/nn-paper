"""
LSTM + Attention 모델 (PyTorch)
- 60일 시퀀스 → 3-class 분류 (하락/보합/상승)
- ~30K params (GTX 950 2GB OK)
- Attention mechanism for temporal weighting
"""
import logging
import math
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LSTMModel:

    def __init__(self, params: dict = None):
        from nn_config import LSTM_PARAMS
        self.params = params or LSTM_PARAMS.copy()
        self.model = None
        self.device = None
        self._setup_device()

    def _setup_device(self):
        try:
            import torch
            if self.params.get("device") == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("LSTM using GPU: %s", torch.cuda.get_device_name(0))
            else:
                self.device = torch.device("cpu")
                logger.info("LSTM using CPU")
        except ImportError:
            self.device = None
            logger.warning("PyTorch not available — LSTM disabled")

    def _build_model(self, input_size: int):
        import torch
        import torch.nn as nn

        class AttentionLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers,
                         dropout, num_classes, use_attention=True):
                super().__init__()
                self.hidden_size = hidden_size
                self.use_attention = use_attention

                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                )

                if use_attention:
                    self.attention = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.Tanh(),
                        nn.Linear(hidden_size // 2, 1),
                    )

                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                # x: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)

                if self.use_attention:
                    attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
                    attn_weights = torch.softmax(attn_weights, dim=1)
                    context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)
                else:
                    context = lstm_out[:, -1, :]  # last timestep

                out = self.dropout(context)
                out = self.fc(out)
                return out

        self.params["input_size"] = input_size  # 실제 피처 수 저장
        self.model = AttentionLSTM(
            input_size=input_size,
            hidden_size=self.params["hidden_size"],
            num_layers=self.params["num_layers"],
            dropout=self.params["dropout"],
            num_classes=self.params["num_classes"],
            use_attention=self.params.get("attention", True),
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info("LSTM model: %d params (%.1f KB)", param_count, param_count * 4 / 1024)

    def train(self, X_seq: np.ndarray, y: np.ndarray,
              X_val_seq: np.ndarray = None, y_val: np.ndarray = None,
              sample_weight: np.ndarray = None) -> dict:
        """
        LSTM 학습.
        X_seq: (n_samples, seq_len, n_features)
        y: (n_samples,)
        """
        if self.device is None:
            return {"error": "PyTorch not available"}

        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        input_size = X_seq.shape[2]
        self._build_model(input_size)

        X_t = torch.FloatTensor(X_seq).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)

        if sample_weight is not None:
            w_t = torch.FloatTensor(sample_weight).to(self.device)
        else:
            w_t = None

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=self.params["batch_size"],
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.params["epochs"]):
            self.model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            # Validation
            val_loss = avg_loss
            if X_val_seq is not None:
                self.model.eval()
                with torch.no_grad():
                    X_v = torch.FloatTensor(X_val_seq).to(self.device)
                    y_v = torch.LongTensor(y_val).to(self.device)
                    v_out = self.model(X_v)
                    val_loss = criterion(v_out, y_v).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.params["patience"]:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d: loss=%.4f, val_loss=%.4f", epoch + 1, avg_loss, val_loss)

        if best_state:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        # 결과 (배치 예측으로 OOM 방지)
        train_pred = self._predict_batched(X_seq, return_proba=False)
        train_acc = np.mean(train_pred == y)

        result = {"train_accuracy": float(train_acc), "best_val_loss": float(best_val_loss)}
        if X_val_seq is not None:
            val_pred = self._predict_batched(X_val_seq, return_proba=False)
            val_acc = np.mean(val_pred == y_val)
            result["val_accuracy"] = float(val_acc)

        logger.info("LSTM 학습 완료: train_acc=%.4f", train_acc)
        return result

    def _predict_batched(self, X_seq: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """GPU OOM 방지를 위한 배치 예측."""
        import torch
        self.model.eval()
        batch_size = 512
        results = []
        with torch.no_grad():
            for i in range(0, len(X_seq), batch_size):
                batch = torch.FloatTensor(X_seq[i:i + batch_size]).to(self.device)
                logits = self.model(batch)
                if return_proba:
                    results.append(torch.softmax(logits, dim=1).cpu().numpy())
                else:
                    results.append(logits.argmax(dim=1).cpu().numpy())
                del batch
        return np.concatenate(results)

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("LSTM 모델이 준비되지 않음")
        return self._predict_batched(X_seq, return_proba=False)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("LSTM 모델이 준비되지 않음")
        return self._predict_batched(X_seq, return_proba=True)

    def save(self, path: str):
        if self.model is None:
            return
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "params": self.params,
        }, path)
        logger.info("LSTM 모델 저장: %s", path)

    def load(self, path: str):
        if self.device is None:
            logger.warning("PyTorch unavailable, cannot load LSTM")
            return
        import torch
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.params = data["params"]
        input_size = self.params["input_size"]
        self._build_model(input_size)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
        logger.info("LSTM 모델 로드: %s", path)
