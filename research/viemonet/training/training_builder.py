from transformers import TrainingArguments, EarlyStoppingCallback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import json
import os

from viemonet.models.model import ViemonetModel
from viemonet.training.callbacks import TrainAccuracyCallback
from viemonet.config import config, device
from viemonet.constant import METHOD
from viemonet.training.data_collator import EmotionDataCollator
from viemonet.training.two_group_trainer import TwoGroupTrainer


class TrainingBuilder:
    def __init__(self, method, head_name, foundation_model_name):
        label_smoothing = config.model.loss.label_smoothing
        self.method = method
        self.head_name = head_name
        self.foundation_model_name = foundation_model_name
        self.trainer = None  # Store trainer for evaluation
        self.test_dataset = None  # Store test dataset
        self.model = ViemonetModel(
            method=method,
            head_model_name=head_name, 
            foundation_model_name=foundation_model_name,
            label_smoothing=label_smoothing
        )

        self.model = self.model.to(device)
        self.training_args = self._build_training_args()
        
    def _build_training_args(self):
        output_dir = config.training_setting.output_dir
        num_train_epochs = config.training_setting.epochs
        per_device_train_batch_size = config.training_setting.batch_train
        per_device_eval_batch_size = config.training_setting.batch_eval
        weight_decay = config.training_setting.weight_decay
        warmup_ratio = config.training_setting.warmup_ratio
        warmup_steps = config.training_setting.warmup_steps
        fp16 = config.training_setting.fp16
        max_grad_norm = config.training_setting.max_grad_norm
        evaluation_strategy = 'epoch'
        save_strategy = 'epoch'

        return TrainingArguments(
            output_dir=f"{output_dir}/{self.method}/UIT-VSMEC/{self.foundation_model_name}/{self.head_name}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            eval_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            fp16=fp16,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type='cosine',
            save_total_limit=3,
            remove_unused_columns=False,  # CRITICAL: Don't remove our custom columns!
        )
    
    def compute_metrics(self, eval_pred):
        """
        Compute accuracy metric for evaluation.
        
        Args:
            eval_pred: EvalPrediction object with predictions and label_ids
            
        Returns:
            Dictionary with accuracy metric
        """
        predictions, labels = eval_pred
        # predictions can be logits (batch_size, num_classes) or dict
        # Handle case where predictions might be a tuple/dict
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Get predicted class (argmax along class dimension)
        if len(predictions.shape) > 1:
            preds = np.argmax(predictions, axis=-1)
        else:
            preds = predictions
        
        # Flatten labels if needed
        labels = labels.flatten() if len(labels.shape) > 1 else labels
        preds = preds.flatten() if len(preds.shape) > 1 else preds
        
        # Calculate accuracy
        accuracy = (preds == labels).astype(np.float32).mean()
        return {"accuracy": float(accuracy)}

    def fit(self, train_data, val_data):
        assert train_data is not None, "Training data must be provided."
        assert val_data is not None, "Validation data must be provided."
        
        # Create custom data collator for handling emotions
        data_collator = EmotionDataCollator()
        
        # Create accuracy callback for tracking accuracy metric
        accuracy_callback = TrainAccuracyCallback()
        
        # Create early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config.training_setting.early_stopping.patience,
            early_stopping_threshold=0.0  # Any improvement counts
        )
        
        self.trainer = TwoGroupTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            callbacks=[accuracy_callback, early_stopping_callback],
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        self.trainer.train()
    
    def evaluate(self, test_data=None):
        """
        Evaluate the model on test data and save results.
        
        Args:
            test_data: Test dataset (EmotionDataset)
        
        Saves:
            - classification_report.txt: Detailed classification metrics
            - classification_report.json: Classification metrics in JSON format
            - evaluation_metrics.png: Loss and accuracy plots
            - evaluation_results.json: Final metrics summary
        """
        assert self.trainer is not None, "Model must be trained before evaluation. Call fit() first."
        
        if test_data is not None:
            self.test_dataset = test_data
        
        assert self.test_dataset is not None, "Test data must be provided for evaluation."
        
        print("\n" + "="*80)
        print("EVALUATING MODEL ON TEST SET")
        print("="*80)
        
        # Get predictions on test set
        predictions_output = self.trainer.predict(self.test_dataset)
        predictions = predictions_output.predictions
        labels = predictions_output.label_ids
        metrics = predictions_output.metrics
        
        # Get predicted classes
        if len(predictions.shape) > 1:
            preds = np.argmax(predictions, axis=-1)
        else:
            preds = predictions
        
        # Generate classification report
        class_names = ['Negative', 'Neutral', 'Positive']
        report_dict = classification_report(
            labels, 
            preds, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        report_str = classification_report(
            labels, 
            preds, 
            target_names=class_names,
            zero_division=0
        )
        
        # Create output directory
        output_dir = self.training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save classification report (text)
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Classification Report - Test Set\n")
            f.write(f"Method: {self.method}\n")
            f.write(f"Foundation Model: {self.foundation_model_name}\n")
            f.write(f"Head Model: {self.head_name}\n")
            f.write("="*80 + "\n\n")
            f.write(report_str)
            f.write("\n\n" + "="*80 + "\n")
            f.write("Test Metrics:\n")
            f.write(f"  Test Loss: {metrics.get('test_loss', 'N/A'):.4f}\n")
            f.write(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Saved classification report to: {report_path}")
        
        # Save classification report (JSON)
        report_json_path = os.path.join(output_dir, 'classification_report.json')
        with open(report_json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"✓ Saved classification report (JSON) to: {report_json_path}")
        
        # Save evaluation results summary
        eval_results = {
            'method': self.method,
            'foundation_model': self.foundation_model_name,
            'head_model': self.head_name,
            'test_loss': float(metrics.get('test_loss', 0)),
            'test_accuracy': float(metrics.get('test_accuracy', 0)),
            'test_samples': len(labels),
            'classification_metrics': report_dict
        }
        
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"✓ Saved evaluation results to: {results_path}")
        
        # Create visualization plots
        self._plot_evaluation_metrics(output_dir, metrics, report_dict)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(report_str)
        print(f"\nTest Loss: {metrics.get('test_loss', 'N/A'):.4f}")
        print(f"Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
        print("="*80 + "\n")
        
        return eval_results
    
    def _plot_evaluation_metrics(self, output_dir, metrics, report_dict):
        """
        Create and save evaluation plots.
        
        Args:
            output_dir: Directory to save plots
            metrics: Metrics dictionary from trainer
            report_dict: Classification report dictionary
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Evaluation Metrics - {self.method}/{self.foundation_model_name}/{self.head_name}', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Per-class Precision, Recall, F1-Score
        ax1 = axes[0, 0]
        classes = ['Negative', 'Neutral', 'Positive']
        precision = [report_dict[str(i)]['precision'] for i in range(3)]
        recall = [report_dict[str(i)]['recall'] for i in range(3)]
        f1 = [report_dict[str(i)]['f1-score'] for i in range(3)]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax1.bar(x - width, precision, width, label='Precision', color='skyblue')
        ax1.bar(x, recall, width, label='Recall', color='lightcoral')
        ax1.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
        
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.set_title('Per-Class Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(classes)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.0])
        
        # Plot 2: Support (samples per class)
        ax2 = axes[0, 1]
        support = [report_dict[str(i)]['support'] for i in range(3)]
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        ax2.bar(classes, support, color=colors)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title('Class Distribution in Test Set')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(support):
            ax2.text(i, v + max(support)*0.02, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Overall Metrics Comparison
        ax3 = axes[1, 0]
        overall_metrics = {
            'Accuracy': report_dict['accuracy'],
            'Macro Avg\nPrecision': report_dict['macro avg']['precision'],
            'Macro Avg\nRecall': report_dict['macro avg']['recall'],
            'Macro Avg\nF1-Score': report_dict['macro avg']['f1-score'],
            'Weighted Avg\nF1-Score': report_dict['weighted avg']['f1-score']
        }
        
        metric_names = list(overall_metrics.keys())
        metric_values = list(overall_metrics.values())
        colors_overall = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
        
        bars = ax3.barh(metric_names, metric_values, color=colors_overall)
        ax3.set_xlabel('Score')
        ax3.set_title('Overall Performance Metrics')
        ax3.set_xlim([0, 1.0])
        ax3.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax3.text(val + 0.02, i, f'{val:.3f}', va='center', fontweight='bold')
        
        # Plot 4: Test Loss and Accuracy
        ax4 = axes[1, 1]
        metrics_summary = {
            'Test Loss': metrics.get('test_loss', 0),
            'Test Accuracy': metrics.get('test_accuracy', 0)
        }
        
        # Create two different y-axes for loss and accuracy
        ax4_twin = ax4.twinx()
        
        bar1 = ax4.bar(0, metrics_summary['Test Loss'], width=0.4, 
                       label='Test Loss', color='#FF5252', alpha=0.7)
        bar2 = ax4_twin.bar(1, metrics_summary['Test Accuracy'], width=0.4, 
                            label='Test Accuracy', color='#4CAF50', alpha=0.7)
        
        ax4.set_ylabel('Loss', color='#FF5252', fontweight='bold')
        ax4_twin.set_ylabel('Accuracy', color='#4CAF50', fontweight='bold')
        ax4.set_title('Test Loss & Accuracy')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Loss', 'Accuracy'])
        ax4.tick_params(axis='y', labelcolor='#FF5252')
        ax4_twin.tick_params(axis='y', labelcolor='#4CAF50')
        ax4_twin.set_ylim([0, 1.0])
        
        # Add value labels
        ax4.text(0, metrics_summary['Test Loss'] + 0.05, 
                f"{metrics_summary['Test Loss']:.4f}", 
                ha='center', va='bottom', fontweight='bold', color='#FF5252')
        ax4_twin.text(1, metrics_summary['Test Accuracy'] + 0.02, 
                     f"{metrics_summary['Test Accuracy']:.4f}", 
                     ha='center', va='bottom', fontweight='bold', color='#4CAF50')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(output_dir, 'evaluation_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved evaluation plots to: {plot_path}")