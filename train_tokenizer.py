import sys, yaml
from bekami.data import iter_corpus
from bekami.tokenizer import train_tokenizer

with open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml") as f:
    c = yaml.safe_load(f)

def lines():
    found = False
    for _, _, files in iter_corpus():
        for _, text in files:
            found = True
            yield from (l for l in text.splitlines() if l.strip())
    if not found:
        raise SystemExit("Ajoute des dépôts dans data/raw/repos/<nom>/ ou des .txt dans data/raw/")

train_tokenizer(lines(), prefix=c["tokenizer_model"], vocab_size=c["vocab_size"])
print(f"Tokenizer créé : {c['tokenizer_model']}.model")
