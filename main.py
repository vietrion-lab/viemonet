import sys
from pathlib import Path

# Add research path
sys.path.append(str(Path(__file__).parent / "research"))

from emotiment.training import Trainer
from emotiment.constant import GRID_MODE, EMOJI2DESCRIPTION_METHOD

if __name__ == "__main__":
    trainer = Trainer(mode=GRID_MODE, method=EMOJI2DESCRIPTION_METHOD)
    trainer.train()
    trainer.evaluate()
    trainer.save_results()
