from emotiment.dataset import tweet_dataset
from emotiment.preprocess import TrainingPreprocessor
from emotiment.config import config
from emotiment.training import TrainingPipeline
from emotiment.constant import GRID_MODE, MONO_MODE, MODEL_LIST
from emotiment.constant.training_constant import EMOJI2VEC_METHOD
import os


class Trainer:
    def __init__(self, mode, method, head_name=None):
        self.mode_matrix = [GRID_MODE, MONO_MODE]
        self.head_name = head_name
        assert mode in self.mode_matrix, f"Invalid mode: {mode}. Choose from {self.mode_matrix}."
        if mode == MONO_MODE:
            assert head_name is not None, "Model name must be provided for mono mode."
            assert head_name in MODEL_LIST, f"Model {head_name} not supported. Choose from {MODEL_LIST}."
            self.head_name = head_name
        self.mode = mode
        self.method = method

        # Build preprocessor instance so we can retain tokenizer & emoji vectors
        self.preprocessor = TrainingPreprocessor(self.method)
        self.input = self.preprocessor(tweet_dataset(config))
        self.extended_tokenizer = getattr(self.preprocessor, 'tokenizer', None)
        self.projected_emoji_vectors = getattr(self.preprocessor, 'projected_emoji_vectors', {})

        # Determine base output directory; if emoji2vec method, nest inside 'emoji2vec'
        base_out = config.training.output_dir
        if self.method == EMOJI2VEC_METHOD:
            base_out = os.path.join(base_out, 'emoji2vec')
        # Ensure directory exists early (harmless if already exists)
        os.makedirs(base_out, exist_ok=True)

        self.trainer = TrainingPipeline(
            self.input,
            head_name=self.head_name,
            mode=self.mode,
            output_root=base_out,
            tokenizer=self.extended_tokenizer,
            projected_emoji_vectors=self.projected_emoji_vectors,
        )
        self._eval_results = None

    def train(self):
        self.trainer.execute()

    def evaluate(self):
        print(f"Evaluating in {self.mode} mode.")
        self._eval_results = self.trainer.evaluate()
        return self._eval_results

    def save_model(self):
        self.trainer.save_model()

    def save_results(self):
        if self._eval_results is None:
            raise ValueError("No evaluation results. Call evaluate() first.")
        self.trainer.save_results(self._eval_results)