import torch
from viemonet.config import config, device
from viemonet.preprocess.preprocessor import TrainingPreprocessor 
from viemonet.preprocess.emotion_dataset import EmotionDataset
from viemonet.training.training_builder import TrainingBuilder
from viemonet.models.main_model_manager import MainModelManager


class Trainer:
    def __init__(
            self, 
            method='separate_emotion',
            model_name='viemonet_phobert',
            dataset_name='uit-vsmec'
        ):
        self.method = method
        self.model_name = model_name
        self.foundation_model = MainModelManager().get_foundation_model_name(model_name)
        self.dataset_name = dataset_name

    def train(self, raw_data=None):
        assert raw_data is not None, "No raw data provided for training."

        pipeline = TrainingPreprocessor(
            method=self.method, 
            model_name=self.model_name, 
            foundation_model_name=self.foundation_model
        )
        comment_emotion_pairs = pipeline(raw_data)

        print(f"Training with method={self.method}, foundation_model={self.foundation_model}, head_name=CNN")

        train_dataset = EmotionDataset(comment_emotion_pairs['train'])
        val_dataset = EmotionDataset(comment_emotion_pairs['validation'])
        test_dataset = EmotionDataset(comment_emotion_pairs['test'])
        
        # Handle imbalanced classes with class weights
        # Calculate class weights based on the training data
        labels_tensor = train_dataset.labels
        num_classes = 3  # Negative (0), Neutral (1), Positive (2)
        class_counts = torch.bincount(labels_tensor, minlength=num_classes).float()
        total = len(labels_tensor)
        class_weights = total / (num_classes * class_counts)
        
        model_builder = TrainingBuilder(
            model_name=self.model_name,
            head_name='cnn',
            foundation_model_name=self.foundation_model,
            dataset_name=self.dataset_name,
            method=self.method,
            class_weights=class_weights
        )
        
        # Store test dataset for evaluation
        model_builder.test_dataset = test_dataset
        
        model_builder.fit(
            train_data=train_dataset,
            val_data=val_dataset
        )
        
        # Evaluate on test set
        model_builder.evaluate()