"""Minimal Hugging Face wrapper used by generation and ICL inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GENERATION_SETTINGS, model_family
from utils import batched, set_seed


class PaperLLM:
    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model_id = model_id
        self.family = model_family(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.yes_token_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.family == "qwen":
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def score_yes_no(self, prompts: Sequence[str], batch_size: int = 8) -> list[bool]:
        """Compare first-token yes/no logits, using yes on an exact tie."""

        decisions: list[bool] = []
        texts = [self._chat_text(prompt) for prompt in prompts]
        for batch in batched(texts, batch_size):
            inputs = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
            logits = output.logits[:, -1, [self.yes_token_id, self.no_token_id]]
            decisions.extend((logits[:, 0] >= logits[:, 1]).cpu().tolist())
        return decisions

    def generate(
        self,
        prompts: Sequence[str],
        batch_size: int = 8,
        max_new_tokens: int = 1000,
        seed: int = 42,
        sample: bool = True,
    ) -> list[str]:
        set_seed(seed)
        texts = [self._chat_text(prompt) for prompt in prompts]
        responses: list[str] = []
        settings = GENERATION_SETTINGS[self.family]
        for batch in batched(texts, batch_size):
            inputs = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(sample and settings.do_sample),
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if generation_kwargs["do_sample"]:
                generation_kwargs.update(
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    top_k=settings.top_k,
                )
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_kwargs)
            continuations = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs.input_ids, generated)
            ]
            responses.extend(
                self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
            )
        return [response.strip() for response in responses]

