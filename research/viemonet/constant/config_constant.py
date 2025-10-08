from pathlib import Path

# Single Source of Truth: Point to root config.yaml
TRAINING_CONFIG_PATH = str(
	Path(__file__).resolve().parents[1] / "config/config_files/default_config.yaml"
)	