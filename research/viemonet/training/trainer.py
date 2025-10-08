from viemonet.config import config, device
from viemonet.preprocess.preprocessor import TrainingPreprocessor 
from viemonet.preprocess.emotion_dataset import EmotionDataset
from viemonet.training.training_builder import TrainingBuilder


class Trainer:
    def __init__(
            self, 
            method='emotion',
            foundation_models=['phobert'], 
            head_names=['lstm']
        ):
        self.head_names = head_names
        self.method = method
        self.foundation_models = foundation_models
        
    def train(self, raw_data=None):
        assert raw_data is not None, "No raw data provided for training."
        # Use little dataset range for testing errors

        for foundation_model in self.foundation_models:
            pipeline = TrainingPreprocessor(method=self.method, foundation_model_name=foundation_model)
            comment_emotion_pairs = pipeline(raw_data)
            
            for head_name in self.head_names:
                print(f"Training with method={self.method}, foundation_model={foundation_model}, head_name={head_name}")

                train_dataset = EmotionDataset(comment_emotion_pairs['train'])
                val_dataset = EmotionDataset(comment_emotion_pairs['validation'])
                test_dataset = EmotionDataset(comment_emotion_pairs['test'])
                
                model_builder = TrainingBuilder(
                    method=self.method,
                    head_name=head_name,
                    foundation_model_name=foundation_model
                )
                
                # Store test dataset for evaluation
                model_builder.test_dataset = test_dataset
                
                model_builder.fit(
                    train_data=train_dataset,
                    val_data=val_dataset
                )
                
                # Evaluate on test set
                model_builder.evaluate()