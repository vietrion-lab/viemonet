import sys
from pathlib import Path

# Add research path
sys.path.append(str(Path(__file__).parent / "research"))

from emotiment.training import Trainer
from emotiment.constant import MONO_MODE, EMOJI2DESCRIPTION_METHOD

if __name__ == "__main__":
    # Train single BiGRU model (faster)
    trainer = Trainer(mode=MONO_MODE, method=EMOJI2DESCRIPTION_METHOD, head_name='bigru')
    trainer.train()
    trainer.evaluate()
    trainer.save_results()
