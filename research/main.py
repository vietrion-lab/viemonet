from viemonet import Trainer
from utils import load_dataset, load_config


if __name__ == "__main__":
    config = load_config("../config.yaml")
    dataset = load_dataset(config.datasets.social_comments.path)
    trainer = Trainer(
        method='no_emotion',
        model_name='vit5',
        dataset_name='uit-vsmec-no-emotion'
    )
    trainer.train(raw_data=dataset)