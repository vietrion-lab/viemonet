from datasets import load_dataset

def emoji_dataset(config):
    dataset = load_dataset(f"{config.dataset.author}/{config.dataset.emoji_data_name}", split='train')
    return dataset

