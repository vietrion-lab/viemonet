import os
from pathlib import Path
from typing import Dict

ENV_FILE_NAMES = [".env", ".env.local"]

_loaded = False

def _parse_line(line: str):
    if not line or line.startswith('#'):
        return None, None
    if '=' not in line:
        return None, None
    key, val = line.split('=', 1)
    return key.strip(), val.strip()


def load_env(force: bool = False) -> Dict[str, str]:
    """Load environment variables from a .env file into os.environ.
    Does nothing if already loaded unless force=True.
    Returns dict of variables loaded.
    """
    global _loaded
    if _loaded and not force:
        return {}

    root = Path(__file__).resolve().parents[3]  # go back to repository root
    loaded: Dict[str, str] = {}
    for fname in ENV_FILE_NAMES:
        fpath = root / fname
        if not fpath.exists():
            continue
        for line in fpath.read_text(encoding='utf-8').splitlines():
            key, val = _parse_line(line)
            if key and val and key not in os.environ:
                os.environ[key] = val
                loaded[key] = val
    # Provide compatibility aliases for Hugging Face tokens
    if 'HF_API_KEY' in os.environ and 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_API_KEY']
    if 'HF_TOKEN' in os.environ and 'HUGGING_FACE_HUB_TOKEN' not in os.environ:
        os.environ['HUGGING_FACE_HUB_TOKEN'] = os.environ['HF_TOKEN']
    _loaded = True
    return loaded
