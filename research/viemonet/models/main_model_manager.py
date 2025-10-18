from viemonet.constant.training_constant import MAIN_MODEL_LIST, FOUNDATION_MODEL_LIST
from viemonet.models.main_models.viemonet_phobert import ViemonetModel as ViemonetPhobertModel
from viemonet.models.main_models.viemonet_visobert import ViemonetModel as ViemonetVisobertModel
from viemonet.models.main_models.vit5 import ViT5Model
from viemonet.models.main_models.phobert import PhoBERTModel
from viemonet.models.main_models.visobert import VisoBERTModel
from viemonet.config import config


class MainModelManager:
    def __init__(self):
        self.label_smoothing = config.model.loss.label_smoothing
    
    def get_model(self, model_name: str, class_weights=None):
        assert model_name in MAIN_MODEL_LIST, \
            f"Model {model_name} not recognized. Available models: {MAIN_MODEL_LIST}"
        assert class_weights is not None, "Class weights must be provided."
            
        if model_name == MAIN_MODEL_LIST[0]:
            return ViemonetPhobertModel(class_weights=class_weights, label_smoothing=self.label_smoothing)
        elif model_name == MAIN_MODEL_LIST[1]:
            return ViemonetVisobertModel(class_weights=class_weights, label_smoothing=self.label_smoothing)
        elif model_name == MAIN_MODEL_LIST[2]:
            return PhoBERTModel(class_weights=class_weights, label_smoothing=self.label_smoothing)
        elif model_name == MAIN_MODEL_LIST[3]:
            return VisoBERTModel(class_weights=class_weights, label_smoothing=self.label_smoothing)
        elif model_name == MAIN_MODEL_LIST[4]:
            return ViT5Model(class_weights=class_weights, label_smoothing=self.label_smoothing)

    def get_foundation_model_name(self, model_name: str):
        assert model_name in MAIN_MODEL_LIST, \
            f"Model {model_name} not recognized. Available models: {MAIN_MODEL_LIST}"
            
        if model_name == MAIN_MODEL_LIST[0] or model_name == MAIN_MODEL_LIST[2] \
            or model_name == MAIN_MODEL_LIST[5] or model_name == MAIN_MODEL_LIST[6]:
            return FOUNDATION_MODEL_LIST[0]  # 'phobert'
        elif model_name == MAIN_MODEL_LIST[1] or model_name == MAIN_MODEL_LIST[3]:
            return FOUNDATION_MODEL_LIST[1]  # 'viobert'
        elif model_name == MAIN_MODEL_LIST[4]:
            return FOUNDATION_MODEL_LIST[2]  # 'vit5'