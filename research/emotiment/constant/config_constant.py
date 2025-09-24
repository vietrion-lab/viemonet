from pathlib import Path

# Compute path relative to the repository regardless of current working directory
TRAINING_CONFIG_PATH = str(
	Path(__file__).resolve().parents[1] / "config" / "config_files" / "train_config.yaml"
)