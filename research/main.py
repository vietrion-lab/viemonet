from viemonet import Trainer
from utils import load_dataset, load_config


if __name__ == "__main__":
    config = load_config("../config.yaml")
    dataset = load_dataset(config.datasets.social_comments.path)
    trainer = Trainer(
        method='emotion',
        model='viemonet_phobert',
        dataset_name='uit-vsmec'
    )
    trainer.train(raw_data=dataset)