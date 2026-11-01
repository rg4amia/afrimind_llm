import json
from pathlib import Path

import torch
import yaml

ADAPTERS_DIR = Path("adapters")

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_example(tok, rec, max_length):
    """Un enregistrement {"messages": [...]} ou {"prompt", "completion"} devient (input_ids, labels).
    Seul le dernier tour de l'assistant contribue à la perte. Renvoie None si le prompt seul dépasse max_length."""
    messages = rec.get("messages") or [
        {"role": "user", "content": rec["prompt"]},
        {"role": "assistant", "content": rec["completion"]},
    ]
    assert messages[-1]["role"] == "assistant", "le dernier message doit être celui de l'assistant"
    if tok.chat_template:
        prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(messages, tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("le chat template ne produit pas un préfixe stable ; adapter build_example")
    else:
        prompt = messages[-2]["content"] + "\n"
        full = prompt + messages[-1]["content"] + (tok.eos_token or "")
    ids = tok(full, add_special_tokens=False)["input_ids"]
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    if n_prompt >= max_length:
        return None
    ids = ids[:max_length]
    return ids, [-100] * n_prompt + ids[n_prompt:]

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tok, records, max_length):
        built = [build_example(tok, r, max_length) for r in records]
        self.items = [e for e in built if e is not None]
        self.skipped = len(built) - len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return {"input_ids": self.items[i][0], "labels": self.items[i][1]}

class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        }

def load_base(base_model, quantize="none"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {}
    if quantize == "4bit":
        compute = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute)
        kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    return model, tok

def load_with_adapter(base_model, adapter_dir=None, quantize="none"):
    model, tok = load_base(base_model, quantize)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model.eval(), tok

def merge_adapter(base_model, adapter_dir, out_dir):
    """Fusionne un adaptateur dans le modèle de base (chargé sans quantification) et l'enregistre."""
    model, tok = load_with_adapter(base_model, adapter_dir, quantize="none")
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

def make_hf_complete(model, tok):
    """Même contrat que bekami.evaluate.make_complete : complete(prompt, max_new_tokens) -> texte."""
    def complete(prompt, max_new_tokens=256):
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return complete

def list_adapters(root=ADAPTERS_DIR):
    rows = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / "adapter.yaml").exists()):
        cfg = yaml.safe_load((d / "adapter.yaml").read_text())
        eval_file = d / "eval.json"
        rows.append(dict(
            domain=d.name, base_model=cfg["base_model"],
            trained=(d / "weights" / "adapter_config.json").exists(),
            eval=json.loads(eval_file.read_text()) if eval_file.exists() else {}))
    return rows
