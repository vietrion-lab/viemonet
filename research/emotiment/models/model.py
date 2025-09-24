from transformers import TrainingArguments, EarlyStoppingCallback 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import evaluate

from emotiment.models.two_group_trainer import TwoGroupTrainer
from emotiment.models.cls_head_manager import ClassificationHeadManager
from emotiment.training.decorators import log_step
from emotiment.training.callbacks import TrainAccuracyCallback
from emotiment.config import config
from emotiment.constant import MODEL_LIST, GRID_MODE, MONO_MODE
from emotiment.models.phobert_lora import PhoBERTLoRA


class SenweetModel:
    def __init__(self, mode=None, head_name=None, output_root=None, tokenizer=None, projected_emoji_vectors=None):
        self.mode = mode
        self.head_name = head_name
        # Root directory where all artifacts for this run will be placed.
        self.output_root = output_root or config.training.output_dir
        self.cls_manager = ClassificationHeadManager()
        # Provided tokenizer may already include new emoji tokens
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.foundation_model.tokenizer_name)
        self.override_tokenizer = tokenizer
        self.projected_emoji_vectors = projected_emoji_vectors or {}
        self._per_head_results = {}
        # Custom simple collator (fixed-length already) to avoid tokenizer.pad issues
        def _simple_collator(features):
            import torch as _torch
            input_ids = [f["input_ids"] for f in features]
            attn = [f["attention_mask"] for f in features]
            labels = [f["labels"] for f in features]
            return {
                "input_ids": _torch.tensor(input_ids, dtype=_torch.long),
                "attention_mask": _torch.tensor(attn, dtype=_torch.long),
                "labels": _torch.tensor(labels, dtype=_torch.long),
            }
        self.data_collator = _simple_collator
        self.acc_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")

    def clear_gpu_cache(self):
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": self.acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": self.f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    
    # ---------------- Emoji embedding helpers ----------------
    def _maybe_inject_embeddings(self, head_model):
        """If tokenizer has new emoji tokens, rebuild PhoBERTLoRA to include them."""
        try:
            if not self.override_tokenizer or not hasattr(head_model, 'classifier_encoder'):
                return head_model
            enc = head_model.classifier_encoder
            if not isinstance(enc, PhoBERTLoRA):
                return head_model
            base_embed = enc.model.get_input_embeddings()
            if base_embed.num_embeddings >= len(self.override_tokenizer):
                return head_model  # already resized
            new_enc = PhoBERTLoRA(tokenizer=self.override_tokenizer, projected_emoji_vectors=self.projected_emoji_vectors)
            head_model.classifier_encoder = new_enc
        except Exception as e:
            print(f"[EMOJI] Injection skipped: {e}")
        return head_model

    def _prepare_head(self, head_name):
        model = self.cls_manager.get_model_by_name(head_name)
        return self._maybe_inject_embeddings(model)
            
    def build_training_args(self, output_dir=None):
        training_config = config.training
        out_dir = output_dir or self.output_root
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=training_config.batch_train,
            per_device_eval_batch_size=training_config.batch_eval,
            num_train_epochs=training_config.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=training_config.early_stopping.metric,
            greater_is_better=True,
            weight_decay=training_config.weight_decay,
            warmup_ratio=training_config.warmup_ratio,
            max_grad_norm=training_config.max_grad_norm,
            fp16=training_config.fp16 and torch.cuda.is_available(),
            logging_steps=50,
            save_total_limit=2,
        )
        return args
    
    @log_step
    def training(self, model, training_args, train_data, eval_data):
        # Move model to GPU if available (simple device placement)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)
        print(f"[DEVICE] Model on {device}")

        # ---- Safety diagnostics (common CUDA assert causes) ----
        try:
            # 1. Check tokenizer vs embedding size for each head encoder
            if hasattr(model, 'classifier_encoder'):
                enc = model.classifier_encoder
                if hasattr(enc, 'model') and hasattr(enc.model, 'get_input_embeddings'):
                    emb = enc.model.get_input_embeddings()
                    emb_n, emb_dim = emb.num_embeddings, emb.embedding_dim
                    tok_n = len(self.tokenizer) if self.tokenizer else -1
                    print(f"[EMOJI-CHECK] embedding rows={emb_n} dim={emb_dim} tokenizer size={tok_n}")
                    if tok_n > 0 and emb_n < tok_n:
                        raise RuntimeError(f"Embedding rows ({emb_n}) < tokenizer size ({tok_n}). Emoji tokens not injected.")
            # 2. Check label range in train_data
            import math as _math
            max_label = -1
            for sample in train_data:
                lbl = int(sample['labels'])
                if lbl > max_label:
                    max_label = lbl
            # Infer expected num classes from head output layer if possible
            expected = None
            for attr in dir(model):
                if attr.startswith('classifier_out'):
                    layer = getattr(model, attr)
                    if hasattr(layer, 'out_features'):
                        expected = layer.out_features
                        break
            if expected is not None:
                print(f"[LABEL-CHECK] max_label={max_label} expected_classes={expected}")
                if max_label >= expected:
                    raise RuntimeError(f"Label value {max_label} >= classifier out_dim {expected}")
        except Exception as diag_e:
            print(f"[SAFETY] Pre-train diagnostic warning: {diag_e}")

        trainer = TwoGroupTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping.patience),
                TrainAccuracyCallback(sample_size=256)
            ],
        )
        trainer.train()
    
    @log_step
    def fit(self, train_data, eval_data):
        import os, json, torch

        def _evaluate_and_save(head_name, model, head_dir):
            # Evaluate model on eval_data directly (independent of evaluate())
            import torch as _torch, numpy as _np
            from sklearn.metrics import classification_report as _cls_report
            device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            all_logits, all_labels = [], []
            with _torch.no_grad():
                for sample in eval_data:  # use validation set for immediate feedback
                    input_ids = _torch.tensor(sample['input_ids'], dtype=_torch.long, device=device).unsqueeze(0)
                    attention_mask = _torch.tensor(sample['attention_mask'], dtype=_torch.long, device=device).unsqueeze(0)
                    label = sample['labels']
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    all_logits.append(out['logits'].cpu().numpy())
                    all_labels.append(label)
            logits = _np.concatenate(all_logits, axis=0)
            labels = _np.array(all_labels)
            metrics = self.compute_metrics((logits, labels))
            preds = _np.argmax(logits, axis=-1)
            report = _cls_report(labels, preds, output_dict=True, zero_division=0)
            result_payload = {
                'metrics': metrics,
                'report': report,
                'labels': labels.tolist(),
                'predictions': preds.tolist()
            }
            self._per_head_results[head_name] = result_payload
            # Save evaluation JSON
            with open(os.path.join(head_dir, 'evaluation_results.json'), 'w') as f:
                json.dump(result_payload, f, indent=2)
            # Save learning curve chart for this head
            try:
                self.plot_learning_curves(checkpoints_dir=head_dir, output_path=os.path.join(head_dir, 'accuracy_curve.png'))
            except Exception as e:
                print(f"[PLOT] Skip chart for {head_name}: {e}")
            # Save final lightweight model weights (state_dict)
            torch.save(model.state_dict(), os.path.join(head_dir, 'pytorch_model.bin'))

        # Stash test dataset for later evaluation inside helper
        self._last_test_dataset = eval_data  # temporarily reuse eval_data if test later replaced

        if self.mode == GRID_MODE:
            for head_name in MODEL_LIST:
                head_dir = os.path.join(self.output_root, head_name)
                os.makedirs(head_dir, exist_ok=True)
                training_args = self.build_training_args(output_dir=head_dir)
                model = self._prepare_head(head_name)
                self.training(model, training_args, train_data, eval_data)
                _evaluate_and_save(head_name, model, head_dir)
                # Release model to free GPU memory before next head
                self.cls_manager.release(head_name)
            # Aggregate summary at root output dir
            summary_path = os.path.join(self.output_root, 'summary_results.json')
            import json as _json
            with open(summary_path, 'w') as f:
                _json.dump(self._per_head_results, f, indent=2)
            print(f"[EVAL] Per-head summary saved to {summary_path}")
            # Copy per-head artifacts into a unified folder outputs/all/<head_name>
            try:
                import shutil
                all_dir = os.path.join(self.output_root, 'all')
                os.makedirs(all_dir, exist_ok=True)
                for head_name in MODEL_LIST:
                    src_head = os.path.join(self.output_root, head_name)
                    dst_head = os.path.join(all_dir, head_name)
                    os.makedirs(dst_head, exist_ok=True)
                    for fname in ['evaluation_results.json', 'pytorch_model.bin', 'accuracy_curve.png']:
                        fpath = os.path.join(src_head, fname)
                        if os.path.isfile(fpath):
                            shutil.copy2(fpath, os.path.join(dst_head, fname))
            except Exception as e:
                print(f"[AGG] Skipped building aggregated folder: {e}")
        elif self.mode == MONO_MODE:
            head_name = self.head_name
            head_dir = os.path.join(self.output_root, head_name)
            os.makedirs(head_dir, exist_ok=True)
            training_args = self.build_training_args(output_dir=head_dir)
            model = self._prepare_head(head_name)
            self.training(model, training_args, train_data, eval_data)
            _evaluate_and_save(head_name, model, head_dir)
            # Optional release (if user trains multiple mono runs sequentially)
            self.cls_manager.release(head_name)

    @log_step
    def evaluate(self, test_data):
        """Evaluate trained model(s) on test dataset.

        Returns:
            dict: metrics per head (grid) or single metrics dict (mono).
        """
        import torch, numpy as np
        from sklearn.metrics import classification_report

        if not hasattr(self, '_evaluate_single'):
            def _evaluate_single(model, dataset):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model.eval()
                all_logits, all_labels = [], []
                with torch.no_grad():
                    for sample in dataset:
                        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long, device=device).unsqueeze(0)
                        attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long, device=device).unsqueeze(0)
                        label = sample['labels']
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                        all_logits.append(out['logits'].cpu().numpy())
                        all_labels.append(label)
                logits = np.concatenate(all_logits, axis=0)
                labels = np.array(all_labels)
                metrics = self.compute_metrics((logits, labels))
                preds = np.argmax(logits, axis=-1)
                cls_report = classification_report(labels, preds, output_dict=True, zero_division=0)
                return metrics, cls_report, labels.tolist(), preds.tolist()
            self._evaluate_single = _evaluate_single  # cache

        # If per-head results already computed during fit, just return them
        if self._per_head_results:
            return self._per_head_results

        results = {}
        if self.mode == GRID_MODE:
            for head_name in MODEL_LIST:
                model = self.cls_manager.get_model_by_name(head_name)
                metrics, report, labels, preds = self._evaluate_single(model, test_data)
                results[head_name] = {
                    'metrics': metrics,
                    'report': report,
                    'labels': labels,
                    'predictions': preds
                }
        else:
            model = self.cls_manager.get_model_by_name(self.head_name)
            metrics, report, labels, preds = self._evaluate_single(model, test_data)
            results[self.head_name] = {
                'metrics': metrics,
                'report': report,
                'labels': labels,
                'predictions': preds
            }
        self._per_head_results = results
        return results

    @log_step
    def save_model(self, base_dir=None):
        """Save model weights (including LoRA adapters + classifier head).

        Structure:
            base_dir/
                <head_name>/
                    pytorch_model.bin
                    metadata.json
        """
        import os, json, torch
        base_dir = base_dir or self.output_root
        os.makedirs(base_dir, exist_ok=True)

        def _save_single(model, head_name):
            head_dir = os.path.join(base_dir, head_name)
            os.makedirs(head_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(head_dir, 'pytorch_model.bin'))
            meta = {
                'head_name': head_name,
                'mode': self.mode,
                'frozen_backbone': True,
                'lora_only_trainable': True,
                'classifier_prefix': 'classifier'
            }
            with open(os.path.join(head_dir, 'metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2)

        if self.mode == GRID_MODE:
            for head_name in MODEL_LIST:
                model = self._prepare_head(head_name)
                _save_single(model, head_name)
        else:
            model = self._prepare_head(self.head_name)
            _save_single(model, self.head_name)

    @log_step
    def save_results(self, results, base_dir=None):
        import os, json
        base_dir = base_dir or self.output_root
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, 'evaluation_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[EVAL] Results saved to {out_path}")
        # Generate plot after saving
        try:
            self.plot_learning_curves(checkpoints_dir=base_dir)
        except Exception as e:
            print(f"[PLOT] Skipped plotting due to error: {e}")

    @log_step
    def plot_learning_curves(self, checkpoints_dir=None, output_path=None):
        """Generate accuracy curve (eval accuracy over epochs) from trainer_state.json.

        Looks for latest checkpoint's trainer_state.json under output_dir.
        Saves PNG to output_path (defaults to output_dir/accuracy_curve.png)
        """
        import os, json, glob
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('[PLOT] matplotlib not installed; skipping curve generation.')
            return None

        base_dir = checkpoints_dir or self.output_root
        pattern = os.path.join(base_dir, 'checkpoint-*', 'trainer_state.json')
        files = sorted(glob.glob(pattern))
        if not files:
            print('[PLOT] No trainer_state.json files found.')
            return None
        # Use the last (latest) trainer_state.json
        state_file = files[-1]
        with open(state_file, 'r') as f:
            state = json.load(f)
        history = state.get('log_history', [])
        epochs = [h['epoch'] for h in history if 'eval_accuracy' in h]
        eval_acc = [h['eval_accuracy'] for h in history if 'eval_accuracy' in h]
        eval_f1 = [h.get('eval_macro_f1') for h in history if 'eval_accuracy' in h]
        train_points = [(h['epoch'], h['train_accuracy']) for h in history if 'train_accuracy' in h]

        if not epochs:
            print('[PLOT] No evaluation accuracy logs to plot.')
            return None

        plt.figure(figsize=(7,4))
        plt.plot(epochs, eval_acc, marker='o', label='Eval Accuracy')
        if any(f1 is not None for f1 in eval_f1):
            plt.plot(epochs, eval_f1, marker='x', label='Eval Macro F1')
        if train_points:
            from collections import OrderedDict
            agg = OrderedDict()
            for ep, val in train_points:
                agg[ep] = val  # keep last measurement per epoch
            tr_epochs = list(agg.keys())
            tr_acc = list(agg.values())
            plt.plot(tr_epochs, tr_acc, marker='s', label='Train Accuracy')
        # Derive head/model name from directory basename if available
        head_candidate = os.path.basename(base_dir.rstrip('/'))
        from emotiment.constant import MODEL_LIST as _ALL_HEADS
        if head_candidate in _ALL_HEADS:
            title = f'{head_candidate} - Evaluation Metrics over Epochs'
        else:
            title = 'Evaluation Metrics over Epochs'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_png = output_path or os.path.join(base_dir, 'accuracy_curve.png')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f'[PLOT] Saved learning curve to {out_png}')
        return out_png