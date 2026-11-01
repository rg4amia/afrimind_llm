from pathlib import Path
import sentencepiece as spm

def train_tokenizer(input_files, prefix="tokenizer/beksam", vocab_size=16000):
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=",".join(map(str, input_files)),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        byte_fallback=True,
        shuffle_input_sentence=True,
        input_sentence_size=1000000
    )
    return prefix + ".model"

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
    def encode(self, text):
        return self.sp.encode(text, out_type=int)
    def decode(self, ids):
        return self.sp.decode(list(map(int, ids)))
    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
