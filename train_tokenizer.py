from pathlib import Path
from bekami.tokenizer import train_tokenizer

files = list(Path("data/raw").glob("*.txt"))
if not files:
    raise SystemExit("Ajoute des fichiers .txt dans data/raw/")
train_tokenizer(files, vocab_size=16000)
print("Tokenizer créé : tokenizer/bekami.model")
