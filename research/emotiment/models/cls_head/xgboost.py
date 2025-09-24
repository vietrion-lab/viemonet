import torch
import torch.nn as nn
from emotiment.models.cls_head.base_head import BaseHead
from emotiment.models.phobert_lora import PhoBERTLoRA
from emotiment.config import config

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:  # optional dependency
    _HAS_XGB = False

class XGBoostClassifier(BaseHead):
    """Hybrid head: freeze encoder forward pass for feature extraction then XGBoost.

    Training strategy:
      - During forward (when labels provided) we collect CLS-like mean pooled features.
      - We buffer features/labels until `finalize()` is called (outside of HF Trainer scope).
      - For simplicity inside current Trainer flow we perform a batched pseudo-linear layer
        using a not-fitted placeholder; true XGBoost training would require a custom loop.

    NOTE: This stub allows inclusion in MODEL_LIST without breaking. Real training
    integration can be added later.
    """
    _keys_to_ignore_on_save = []
    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.classifier_encoder = PhoBERTLoRA()
        self._features = []
        self._targets = []
        self.out_dim = 3  # fixed from dataset
        self._booster = None if _HAS_XGB else None
        # Always prepare a linear projection (used as proxy or fallback)
        self.linear_proxy = nn.Linear(768, self.out_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
        x = self.pool(x, attention_mask)  # (B,768)
        # Ensure proxy layer is on same device
        if self.linear_proxy.weight.device != x.device:
            self.linear_proxy = self.linear_proxy.to(x.device)

        if _HAS_XGB and self._booster is not None:
            dm = xgb.DMatrix(x.detach().cpu().numpy())
            preds = self._booster.predict(dm)
            if preds.ndim == 1:
                preds = preds[:, None]
            logits = torch.tensor(preds, device=x.device)
        else:
            # Use proxy linear during training / fallback scenario
            logits = self.linear_proxy(x)
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}

    def collect_for_xgb(self, input_ids, attention_mask, labels):
        if not _HAS_XGB:
            return
        with torch.no_grad():
            x = self.classifier_encoder(input_ids, attention_mask).last_hidden_state
            x = self.pool(x, attention_mask)
            self._features.append(x.cpu())
            self._targets.append(labels.cpu())

    def finalize_xgb(self):
        if not _HAS_XGB:
            return None
        import torch
        if not self._features:
            return None
        X = torch.cat(self._features, dim=0).numpy()
        y = torch.cat(self._targets, dim=0).numpy()
        dm = xgb.DMatrix(X, label=y)
        cfg = config.model.xgboost
        params = {
            'max_depth': cfg.max_depth,
            'eta': cfg.learning_rate,
            'subsample': cfg.subsample,
            'colsample_bytree': cfg.colsample_bytree,
            'objective': 'multi:softprob',
            'num_class': self.out_dim,
            'eval_metric': 'mlogloss'
        }
        self._booster = xgb.train(params, dm, num_boost_round=cfg.n_estimators)
        return self._booster
