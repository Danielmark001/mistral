import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
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

MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


class NOVAPlanner:
    def __init__(self, model_name=MODEL_NAME, device=None, lora_config=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoProcessor.from_pretrained(model_name)
        # keep tokenizer as alias for training compatibility
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
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

    def _build_text_prompt(self, serialized_state: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a navigation agent. Given the current state, output exactly one action word.",
            },
            {
                "role": "user",
                "content": serialized_state,
            },
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _build_vision_prompt(self, task: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": task},
                ],
            }
        ]

    def predict(self, serialized_state: str) -> str:
        """Predict next action from structured text state (used during LoRA training)."""
        prompt = self._build_text_prompt(serialized_state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]

            mask = torch.full_like(logits, float("-inf"))
            for tok_id in self._action_token_ids.values():
                mask[:, tok_id] = logits[:, tok_id]

            chosen_id = mask.argmax(dim=-1).item()

        id_to_action = {v: k for k, v in self._action_token_ids.items()}
        return id_to_action.get(chosen_id, "move_forward")

    def predict_from_image(self, image: Image.Image, task: str) -> str:
        """
        Predict an action from a real screenshot using vision.
        Returns the model's raw text response (not restricted to action vocab).
        """
        messages = self._build_vision_prompt(task)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def load(cls, path: str, base_model_name=MODEL_NAME, device=None):
        planner = cls.__new__(cls)
        planner.model_name = base_model_name
        planner.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        planner.processor = AutoProcessor.from_pretrained(path)
        planner.tokenizer = planner.processor.tokenizer
        planner.tokenizer.pad_token = planner.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if planner.device == "cuda" else torch.float32,
            device_map="auto" if planner.device == "cuda" else None,
        )
        planner.model = PeftModel.from_pretrained(base_model, path)
        if planner.device != "cuda":
            planner.model = planner.model.to(planner.device)

        planner._action_token_ids = planner._get_action_token_ids()
        return planner
