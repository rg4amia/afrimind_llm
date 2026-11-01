import argparse, json, random, shutil
from pathlib import Path
import numpy as np, torch, yaml
from transformers import Trainer, TrainingArguments
from bekami.hf import ADAPTERS_DIR, PadCollator, SFTDataset, load_base, load_jsonl

ap = argparse.ArgumentParser(description="Fine-tuning QLoRA d'un adaptateur de domaine")
ap.add_argument("domain", help="dossier adapters/<domaine>/ contenant adapter.yaml et les données")
ap.add_argument("--max-steps", type=int, default=-1, help="limite le nombre de pas (test rapide)")
args = ap.parse_args()

root = ADAPTERS_DIR / args.domain
c = yaml.safe_load((root / "adapter.yaml").read_text())
random.seed(c["seed"]); np.random.seed(c["seed"]); torch.manual_seed(c["seed"])

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model, tok = load_base(c["base_model"], c["quantize"])
if c["quantize"] == "4bit":
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(
    r=c["lora_r"], lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
    target_modules=c["target_modules"], task_type="CAUSAL_LM", bias="none"))
model.print_trainable_parameters()

train = SFTDataset(tok, load_jsonl(root / c["train_file"]), c["max_length"])
val = SFTDataset(tok, load_jsonl(root / c["val_file"]), c["max_length"]) if c.get("val_file") else None
print(f"{len(train)} exemples d'entraînement ({train.skipped} écartés : prompt trop long)")
if len(train) == 0:
    raise SystemExit("Aucun exemple exploitable.")

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
trainer = Trainer(
    model=model, train_dataset=train, eval_dataset=val, data_collator=PadCollator(tok.pad_token_id),
    args=TrainingArguments(
        output_dir=str(root / "runs"), per_device_train_batch_size=c["batch_size"],
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        learning_rate=c["learning_rate"], num_train_epochs=c["epochs"], max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=10, save_strategy="no",
        eval_strategy="epoch" if val else "no", bf16=bf16, fp16=cuda and not bf16,
        gradient_checkpointing=cuda, report_to="none", remove_unused_columns=False,
        seed=c["seed"], use_cpu=not cuda))
trainer.train()

out = root / "weights"
model.save_pretrained(out)
tok.save_pretrained(out)
shutil.copy(root / "adapter.yaml", out / "adapter.yaml")
(out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
print("Adaptateur enregistré :", out)
