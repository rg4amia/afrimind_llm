from pathlib import Path
import sentencepiece as spm

# ids 0-3 : pad, unk, bos, eos ; les symboles ci-dessous prennent les ids suivants, dans cet ordre
SPECIAL_TOKENS = [
    "<fim_prefix>", "<fim_middle>", "<fim_suffix>",
    "<file_sep>", "<repo_name>",
    "<tool_call>", "<tool_result>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
]

def train_tokenizer(sentences, prefix="tokenizer/bekami", vocab_size=64000):
    """`sentences` : itérable de chaînes (une ligne de code ou de texte par élément)."""
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(sentences),
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=SPECIAL_TOKENS,
        byte_fallback=True,
        split_digits=True,
        # le code est sensible aux espaces : on ne normalise rien
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        allow_whitespace_only_pieces=True,
        max_sentence_length=16384,
        shuffle_input_sentence=True,
        input_sentence_size=5000000,
    )
    return prefix + ".model"

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.special = {t: self.sp.piece_to_id(t) for t in SPECIAL_TOKENS}
        self.eos_id = self.sp.eos_id()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(list(map(int, ids)))

    def chat(self, messages):
        """messages : liste de (rôle, texte), rôle dans system/user/assistant. Se termine par le tour
        de l'assistant ouvert, prêt pour la génération."""
        ids = []
        for role, text in messages:
            ids += [self.special[f"<|{role}|>"]] + self.encode(text) + [self.special["<|end|>"]]
        return ids + [self.special["<|assistant|>"]]

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()
