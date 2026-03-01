import wandb
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from models.planner import NOVAPlanner
from training.dataset_builder import load_jsonl


class ActionDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        for r in records:
            prompt = r["input"] + "\nAction: " + r["output"]
            self.items.append(prompt)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.items[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train_planner(
    planner: NOVAPlanner,
    train_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    generation: int = 0,
    wandb_project: str = "nova-planner",
    manage_wandb: bool = True,
):
    # if a run is already open (called from self_improve), use it
    if manage_wandb and wandb.run is None:
        wandb.init(
            project=wandb_project,
            name=f"nova-gen-{generation}",
            tags=["training", f"gen-{generation}"],
        )

    records = load_jsonl(train_path)
    dataset = ActionDataset(records, planner.tokenizer)

    wandb.log({"train/dataset_size": len(dataset), "generation": generation})

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        bf16=planner.device == "cuda",
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name=f"nova-gen-{generation}",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=planner.model,
        args=args,
        train_dataset=dataset,
        tokenizer=planner.tokenizer,
    )

    trainer.train()
    planner.save(output_dir)

    if manage_wandb and wandb.run is not None:
        wandb.finish()

    print(f"Training complete. Model saved to {output_dir}")
