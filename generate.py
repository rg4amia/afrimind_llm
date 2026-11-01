import torch
from bekami.model import GPTConfig, BekamiGPT
from bekami.tokenizer import Tokenizer

ckpt = torch.load("checkpoints/bekami_final.pt", map_location="cpu", weights_only=False)
cfg = GPTConfig(**ckpt["config"])
model = BekamiGPT(cfg)
model.load_state_dict(ckpt["model"])
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
tok = Tokenizer("tokenizer/bekami.model")

prompt = input("Prompt: ")
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
out = model.generate(ids, max_new_tokens=150)
print(tok.decode(out[0].tolist()))
