from viemonet.constant import MODEL_LIST
from viemonet.models.cls_head.bigru import BiGRUClassifier
from viemonet.models.cls_head.bilstm import BiLSTMClassifier
from viemonet.models.cls_head.cnn import CNNClassifier
from viemonet.models.cls_head.gru import GRUClassifier
from viemonet.models.cls_head.lstm import LSTMClassifier


class ClassificationHeadManager:
    def __init__(self):
        pass

    def get_model(self, head_name, foundation_model_name):
        assert head_name in MODEL_LIST, f"Unknown model: {head_name}. Must be one of {MODEL_LIST}"
        if head_name == 'lstm':
            return LSTMClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'gru':
            return GRUClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'bigru':
            return BiGRUClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'bilstm':
            return BiLSTMClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'cnn':
            return CNNClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'logreg':
            return LogisticRegressionClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'xgboost':
            return XGBoostClassifier(foundation_model_name=foundation_model_name)