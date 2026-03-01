import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PeftModel
from .lora_config import get_lora_config


ALLOWED_ACTIONS = [
    "turn_left",
    "turn_right",
    "move_forward",
    "pickup",
    "drop",
    "toggle",
    "done",
]


class NOVAPlanner:
    def __init__(self, model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503", device=None, lora_config=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        cfg = lora_config or get_lora_config()
        self.model = get_peft_model(base_model, cfg)

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self._action_token_ids = self._get_action_token_ids()

    def _get_action_token_ids(self):
        ids = {}
        for action in ALLOWED_ACTIONS:
            tokens = self.tokenizer.encode(action, add_special_tokens=False)
            if tokens:
                ids[action] = tokens[0]
        return ids

    def predict(self, serialized_state: str) -> str:
        prompt = serialized_state + "\nAction:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

            # restrict logits to allowed action tokens only
            allowed_ids = list(self._action_token_ids.values())
            mask = torch.full_like(logits, float("-inf"))
            for tok_id in allowed_ids:
                mask[:, tok_id] = logits[:, tok_id]

            chosen_id = mask.argmax(dim=-1).item()

        id_to_action = {v: k for k, v in self._action_token_ids.items()}
        return id_to_action.get(chosen_id, "move_forward")

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, base_model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503", device=None):
        planner = cls.__new__(cls)
        planner.model_name = base_model_name
        planner.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        planner.tokenizer = AutoTokenizer.from_pretrained(path)
        planner.tokenizer.pad_token = planner.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if planner.device == "cuda" else torch.float32,
            device_map="auto" if planner.device == "cuda" else None,
        )
        planner.model = PeftModel.from_pretrained(base_model, path)
        if planner.device != "cuda":
            planner.model = planner.model.to(planner.device)

        planner._action_token_ids = planner._get_action_token_ids()
        return planner
