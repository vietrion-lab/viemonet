from viemonet import Trainer
from utils import load_dataset, load_config


if __name__ == "__main__":
    config = load_config("../config.yaml")
    dataset = load_dataset(config.datasets.vietnamese_comments.path)
    trainer = Trainer(
        method='no_emotion',
        model_name='phobert',
        dataset_name='uit-vsfc'
    )
    trainer.train(raw_data=dataset)