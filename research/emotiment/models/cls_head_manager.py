from emotiment.constant import MODEL_LIST

# Lazy import wrapper functions to avoid allocating all heads at once
def _build_head(name):
    if name == 'lstm':
        from emotiment.models.cls_head.lstm import LSTMClassifier
        return LSTMClassifier()
    if name == 'gru':
        from emotiment.models.cls_head.gru import GRUClassifier
        return GRUClassifier()
    if name == 'bigru':
        from emotiment.models.cls_head.bigru import BiGRUClassifier
        return BiGRUClassifier()
    if name == 'bilstm':
        from emotiment.models.cls_head.bilstm import BiLSTMClassifier
        return BiLSTMClassifier()
    if name == 'cnn':
        from emotiment.models.cls_head.cnn import CNNClassifier
        return CNNClassifier()
    if name == 'logreg':
        from emotiment.models.cls_head.logreg import LogisticRegressionClassifier
        return LogisticRegressionClassifier()
    if name == 'xgboost':
        from emotiment.models.cls_head.xgboost import XGBoostClassifier
        return XGBoostClassifier()
    raise ValueError(f'Unknown classification head: {name}')

class ClassificationHeadManager:
    def __init__(self):
        # Cache only upon first request
        self._cache = {}

    def get_model_by_name(self, head_name):
        if head_name not in self._cache:
            self._cache[head_name] = _build_head(head_name)
        return self._cache[head_name]

    def release(self, head_name: str):
        import gc, torch
        model = self._cache.pop(head_name, None)
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release_all(self):
        for k in list(self._cache.keys()):
            self.release(k)