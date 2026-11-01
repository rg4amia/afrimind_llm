from pathlib import Path
import numpy as np
from tqdm import tqdm
from bekami.tokenizer import Tokenizer

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    sources = sorted(RAW.glob("*.txt"))
    if not sources:
        raise SystemExit("Ajoute d'abord des fichiers .txt dans data/raw/")
    tok = Tokenizer("tokenizer/bekami.model")
    all_ids = []
    for path in sources:
        print("Tokenisation:", path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        ids = tok.encode(text)
        all_ids.extend(ids)
        all_ids.append(3)  # EOS entre documents
    arr = np.asarray(all_ids, dtype=np.uint16)
    split = int(len(arr) * 0.98)
    arr[:split].tofile(OUT / "train.bin")
    arr[split:].tofile(OUT / "val.bin")
    print(f"Tokens: {len(arr):,}; train: {split:,}; validation: {len(arr)-split:,}")

if __name__ == "__main__":
    main()
