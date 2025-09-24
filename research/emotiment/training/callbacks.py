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
                input_ids = torch.tensor(sample["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                attn = torch.tensor(sample["attention_mask"], dtype=torch.long, device=device).unsqueeze(0)
                label = sample["labels"]
                out = model(input_ids=input_ids, attention_mask=attn)
                pred = out["logits"].argmax(-1).item()
                if pred == label:
                    correct += 1
                total += 1
        acc = correct / total if total else 0.0
        trainer.log({"train_accuracy": acc, "epoch": state.epoch})
        return control