from datasets import load_dataset

def tweet_dataset(config):
    dataset = load_dataset(f"{config.dataset.author}/{config.dataset.tweet_data_name}", split='train')
    return dataset