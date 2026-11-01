import math, random, yaml, numpy as np, torch
from pathlib import Path
from tqdm import trange
from beksam.model import GPTConfig, BeksamGPT

with open("config.yaml", "r") as f:
    c = yaml.safe_load(f)

random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

train_data = np.memmap(Path(c["data_dir"])/"train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap(Path(c["data_dir"])/"val.bin", dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data)-c["context_length"]-1, (c["batch_size"],))
    x = torch.stack([torch.from_numpy(data[i:i+c["context_length"]].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+c["context_length"]].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

cfg = GPTConfig(
    vocab_size=c["vocab_size"], context_length=c["context_length"],
    d_model=c["d_model"], n_layers=c["n_layers"], n_heads=c["n_heads"],
    dropout=c["dropout"]
)
model = BeksamGPT(cfg).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters())/1e6, "M")
optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
use_amp = device == "cuda"
amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

def lr_at(step):
    if step < c["warmup_steps"]:
        return c["learning_rate"] * (step+1) / c["warmup_steps"]
    progress = min(1.0, (step-c["warmup_steps"]) / max(1, c["max_steps"]-c["warmup_steps"]))
    return c["min_learning_rate"] + 0.5*(c["learning_rate"]-c["min_learning_rate"])*(1+math.cos(math.pi*progress))

@torch.no_grad()
def estimate():
    model.eval()
    vals = []
    for _ in range(c["eval_batches"]):
        xb,yb = get_batch("val")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb,yb)
        vals.append(loss.item())
    model.train()
    return sum(vals)/len(vals)

Path("checkpoints").mkdir(exist_ok=True)
model.train()
for step in trange(c["max_steps"]):
    lr = lr_at(step)
    for pg in optimizer.param_groups: pg["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    for _ in range(c["gradient_accumulation_steps"]):
        xb,yb = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _, loss = model(xb,yb)
            loss = loss / c["gradient_accumulation_steps"]
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer); scaler.update()

    if step % c["eval_interval"] == 0:
        val_loss = estimate()
        print(f"step {step}: val_loss={val_loss:.4f}, lr={lr:.2e}")
    if step > 0 and step % c["save_interval"] == 0:
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": step},
                   f"checkpoints/step_{step}.pt")

torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": c["max_steps"]},
           "checkpoints/beksam_final.pt")
print("Entraînement terminé.")
