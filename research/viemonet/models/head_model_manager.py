from viemonet.constant import MODEL_LIST
from viemonet.models.cls_head.bigru import BiGRUClassifier
from viemonet.models.cls_head.bilstm import BiLSTMClassifier
from viemonet.models.cls_head.cnn import CNNClassifier
from viemonet.models.cls_head.gru import GRUClassifier
from viemonet.models.cls_head.lstm import LSTMClassifier
from viemonet.models.cls_head.lstm_attention import LSTMAttentionClassifier
from viemonet.models.cls_head.bilstm_attention import BiLSTMAttentionClassifier
from viemonet.models.cls_head.gru_attention import GRUAttentionClassifier
from viemonet.models.cls_head.bigru_attention import BiGRUAttentionClassifier
from viemonet.models.cls_head.transformer_encoder import TransformerEncoderClassifier


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
        if head_name == 'lstm_attention':
            return LSTMAttentionClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'bilstm_attention':
            return BiLSTMAttentionClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'gru_attention':
            return GRUAttentionClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'bigru_attention':
            return BiGRUAttentionClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'transformer_encoder':
            return TransformerEncoderClassifier(foundation_model_name=foundation_model_name)
        if head_name == 'cnn':
            return CNNClassifier(foundation_model_name=foundation_model_name)