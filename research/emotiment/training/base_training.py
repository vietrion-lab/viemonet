from emotiment.models.model import SenweetModel


class TrainingPipeline:
    def __init__(self, tweet_data, head_name=None, mode=None, output_root=None, tokenizer=None, projected_emoji_vectors=None):
        self.tweet_train_data, self.tweet_eval_data, self.tweet_test_data = tweet_data
        # Pass tokenizer & projected emoji vectors so SenweetModel can inject properly
        self.model = SenweetModel(mode, head_name, output_root=output_root, tokenizer=tokenizer, projected_emoji_vectors=projected_emoji_vectors)

    def execute(self):
        self.model.fit(self.tweet_train_data, self.tweet_eval_data)
        return self.model

    def evaluate(self):
        return self.model.evaluate(self.tweet_test_data)

    def save_model(self, base_dir=None):
        self.model.save_model(base_dir)

    def save_results(self, results, base_dir=None):
        self.model.save_results(results, base_dir)