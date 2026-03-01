from peft import LoraConfig, TaskType


def get_lora_config(r=16, lora_alpha=32, lora_dropout=0.1):
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
