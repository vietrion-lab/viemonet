from transformers import TrainerCallback


class TrainAccuracyCallback(TrainerCallback):
    """Callback to compute (sampled) training accuracy at each epoch end.

    For efficiency, you can limit the number of samples evaluated via sample_size.
    """
    def __init__(self, sample_size: int | None = None):
        self.sample_size = sample_size

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[override]
        trainer = kwargs.get("trainer")
        if trainer is None:
            return control
        dataset = trainer.train_dataset
        model = trainer.model
        device = model.device
        model.eval()
        correct = 0
        total = 0
        import torch
        with torch.no_grad():
            for idx in range(len(dataset)):
                if self.sample_size is not None and idx >= self.sample_size:
                    break
                sample = dataset[idx]
                ids = sample["ids"].unsqueeze(0).to(device)
                attn = sample["attn"].unsqueeze(0).to(device)
                emo = [sample["emo"]]  # Wrap in list for batch processing
                label = sample["labels"].item()
                
                out = model(ids=ids, attn=attn, emo=emo)
                pred = out["logits"].argmax(-1).item()
                if pred == label:
                    correct += 1
                total += 1
        acc = correct / total if total else 0.0
        trainer.log({"train_accuracy": acc, "epoch": state.epoch})
        model.train()  # Set back to training mode
        return control