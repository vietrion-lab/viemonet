from viemonet import Trainer
from utils import load_dataset, load_config


if __name__ == "__main__":
    config = load_config("../config.yaml")
    # dataset = load_dataset(config.datasets.social_comments.path)
    dataset = load_dataset(config.datasets.vietnamese_comments.path)
    trainer = Trainer(
        method='describe_emotion',
        model_name='phobert',
        dataset_name='uit-vsfc-ved'
    )
    trainer.train(raw_data=dataset)