MODEL_LIST = [
    'lstm', 'gru', 'bigru', 'bilstm', 'cnn', 
    'lstm_attention', 'bilstm_attention', 'gru_attention', 'bigru_attention', 
    'transformer_encoder',
    'logreg', 'xgboost'
]
FOUNDATION_MODEL_LIST = ['phobert', 'visobert', 'vit5']
METHOD = ['separate_emotion', 'union_emotion', 'no_emotion', 'separate_no_emotion', 'describe_emotion']
MAIN_MODEL_LIST = [
    'viemonet_phobert', 
    'viemonet_no_metacls', 
    'phobert', 
    'visobert', 
    'vit5',
    'viemonet_polarity_comp'
]