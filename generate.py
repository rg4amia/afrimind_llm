import torch
from beksam.model import GPTConfig, BeksamGPT
from beksam.tokenizer import Tokenizer

ckpt = torch.load("checkpoints/beksam_final.pt", map_location="cpu", weights_only=False)
cfg = GPTConfig(**ckpt["config"])
model = BeksamGPT(cfg)
model.load_state_dict(ckpt["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
tok = Tokenizer("tokenizer/beksam.model")

prompt = input("Prompt: ")
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
out = model.generate(ids, max_new_tokens=150)
print(tok.decode(out[0].tolist()))
