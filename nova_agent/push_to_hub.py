"""
Upload trained NOVA LoRA adapters to Hugging Face Hub.

Judging criterion: "We also expect you to save your model as an Artifact
and log to HF for us to validate (e.g. the LoRA adaptors)."

Usage:
    python push_to_hub.py --gen 3 --hf-repo Danielmark001/nova-lora-gen3
    python push_to_hub.py --all --hf-org Danielmark001
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import wandb
from models.planner import NOVAPlanner

HF_ORG = "Danielmark001"
BASE_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
WANDB_PROJECT = "nova-planner"
WANDB_ENTITY = "leadgen12344-nanyang-technological-university-singapore"
OUTPUT_BASE = "outputs/nova"


def push_generation(gen_num: int, hf_repo: str = None, output_base: str = OUTPUT_BASE):
    model_path = Path(output_base) / f"gen_{gen_num}" / "model"
    if not model_path.exists():
        print(f"No model found at {model_path}. Train generation {gen_num} first.")
        sys.exit(1)

    repo_id = hf_repo or f"{HF_ORG}/nova-lora-gen{gen_num}"
    print(f"Loading adapter from {model_path}")
    planner = NOVAPlanner.load(str(model_path), base_model_name=BASE_MODEL)

    print(f"Pushing to HuggingFace: {repo_id}")
    planner.model.push_to_hub(repo_id, commit_message=f"NOVA LoRA adapter — generation {gen_num}")
    planner.processor.push_to_hub(repo_id, commit_message=f"NOVA processor — generation {gen_num}")

    hf_url = f"https://huggingface.co/{repo_id}"
    print(f"Uploaded: {hf_url}")

    # log HF link back to W&B so judges can find it
    if wandb.run is None:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"push-gen-{gen_num}-to-hf",
            tags=["huggingface", f"gen-{gen_num}"],
        )

    artifact = wandb.Artifact(
        name=f"nova-hf-gen-{gen_num}",
        type="model",
        description=f"HuggingFace link for NOVA LoRA gen {gen_num}",
        metadata={
            "hf_repo": repo_id,
            "hf_url": hf_url,
            "generation": gen_num,
            "base_model": BASE_MODEL,
        },
    )
    wandb.log_artifact(artifact)
    wandb.log({"hf/repo": repo_id, "hf/generation": gen_num})
    wandb.finish()
    return hf_url


def push_all(output_base: str = OUTPUT_BASE):
    base = Path(output_base)
    gen_dirs = sorted(base.glob("gen_*"), key=lambda p: int(p.name.split("_")[1]))
    if not gen_dirs:
        print(f"No generation directories found under {base}")
        sys.exit(1)

    urls = []
    for gen_dir in gen_dirs:
        gen_num = int(gen_dir.name.split("_")[1])
        model_path = gen_dir / "model"
        if model_path.exists():
            url = push_generation(gen_num, output_base=output_base)
            urls.append((gen_num, url))

    print("\nAll adapters uploaded:")
    for gen_num, url in urls:
        print(f"  Gen {gen_num}: {url}")


def main():
    parser = argparse.ArgumentParser(description="Push NOVA LoRA adapters to HuggingFace")
    parser.add_argument("--gen", type=int, default=None, help="Generation number to push (e.g. 3)")
    parser.add_argument("--all", action="store_true", help="Push all available generations")
    parser.add_argument("--hf-repo", type=str, default=None, help="Full HF repo ID (overrides default)")
    parser.add_argument("--hf-org", type=str, default=HF_ORG, help="HuggingFace org/username")
    parser.add_argument("--output", type=str, default=OUTPUT_BASE, help="Base output directory")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("Warning: HF_TOKEN not set. Set it to push to a private repo.")

    if args.all:
        push_all(output_base=args.output)
    elif args.gen is not None:
        push_generation(args.gen, hf_repo=args.hf_repo, output_base=args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
