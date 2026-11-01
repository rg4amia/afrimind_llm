import sys, yaml
from pathlib import Path
import numpy as np
from tqdm import tqdm
from bekami.data import iter_corpus, repo_document_ids, is_validation, make_rng
from bekami.tokenizer import Tokenizer

with open(sys.argv[1] if len(sys.argv) > 1 else "config.yaml") as f:
    c = yaml.safe_load(f)

OUT = Path(c["data_dir"])
OUT.mkdir(parents=True, exist_ok=True)
FIM_RATE = c.get("fim_rate", 0.5)

def main():
    tok = Tokenizer(c["tokenizer_model"] + ".model")
    assert tok.vocab_size == c["vocab_size"], "vocab_size du config différent de celui du tokenizer"
    assert tok.vocab_size < 2**16, "tokens stockés en uint16 : vocabulaire limité à 65 535"
    rng = make_rng(c["seed"])
    counts = {"train": 0, "val": 0}
    with open(OUT / "train.bin", "wb") as ftrain, open(OUT / "val.bin", "wb") as fval:
        for name, is_repo, files in tqdm(iter_corpus(), desc="documents"):
            ids = repo_document_ids(tok, name, files, rng, FIM_RATE) if is_repo else \
                tok.encode(files[0][1]) + [tok.eos_id]
            split = "val" if is_validation(name) else "train"
            np.asarray(ids, dtype=np.uint16).tofile(ftrain if split == "train" else fval)
            counts[split] += len(ids)
    if counts["train"] == 0:
        raise SystemExit("Aucun document retenu : vérifie data/raw/")
    if counts["val"] == 0:
        print("Attention : aucun document de validation (peu de dépôts). Ajoute des dépôts ou baisse is_validation.")
    print(f"Tokens: train {counts['train']:,} ; validation {counts['val']:,}")

if __name__ == "__main__":
    main()
