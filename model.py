"""Minimal Hugging Face and vLLM wrappers used by ICL inference."""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GENERATION_SETTINGS, model_family
from utils import batched, set_seed


YesNoProbability = tuple[float, float]


def _label_token_id(tokenizer, label: str) -> int:
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Expected {label!r} to be one token, found {token_ids}")
    return int(token_ids[0])


def _normalized_pair(yes_logprob: float, no_logprob: float) -> YesNoProbability:
    maximum = max(yes_logprob, no_logprob)
    yes = math.exp(yes_logprob - maximum)
    no = math.exp(no_logprob - maximum)
    total = yes + no
    return yes / total, no / total


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
        self.yes_token_id = _label_token_id(self.tokenizer, "yes")
        self.no_token_id = _label_token_id(self.tokenizer, "no")

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.family == "qwen":
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def score_yes_no_probabilities(
        self, prompts: Sequence[str], batch_size: int = 8
    ) -> list[YesNoProbability]:
        """Return probabilities normalized over the first-token yes/no logits."""

        probabilities: list[YesNoProbability] = []
        for prompt_batch in batched(prompts, batch_size):
            batch = [self._chat_text(prompt) for prompt in prompt_batch]
            inputs = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(self.model.device)
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False)
            logits = output.logits[:, -1, [self.yes_token_id, self.no_token_id]]
            batch_probabilities = torch.softmax(logits.float(), dim=-1).cpu().tolist()
            probabilities.extend(
                (float(yes), float(no)) for yes, no in batch_probabilities
            )
        return probabilities

    def score_yes_no(self, prompts: Sequence[str], batch_size: int = 8) -> list[bool]:
        """Compare first-token yes/no probabilities, using yes on an exact tie."""

        return [
            yes >= no
            for yes, no in self.score_yes_no_probabilities(prompts, batch_size)
        ]

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


class VLLMPaperLLM:
    """Offline vLLM backend for one-token yes/no probability scoring."""

    def __init__(
        self,
        model_id: str,
        tensor_parallel_size: int = 1,
        seed: int = 42,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The vLLM backend requires vllm; install requirements.txt"
            ) from exc

        if "logprob_token_ids" not in inspect.signature(SamplingParams).parameters:
            raise RuntimeError(
                "This probability mode requires vLLM >= 0.29; "
                "install the version from requirements.txt"
            )

        self.model_id = model_id
        self.family = model_family(model_id)
        self.model = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            seed=seed,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.sampling_params_class = SamplingParams
        self.yes_token_id = _label_token_id(self.tokenizer, "yes")
        self.no_token_id = _label_token_id(self.tokenizer, "no")

    def _chat_text(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.family == "qwen":
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _logprob(value) -> float:
        return float(value.logprob if hasattr(value, "logprob") else value)

    def score_yes_no_probabilities(
        self, prompts: Sequence[str], batch_size: int = 8
    ) -> list[YesNoProbability]:
        """Pass all prompts to vLLM and return normalized yes/no probabilities."""

        del batch_size  # vLLM schedules the complete request list itself.
        if not prompts:
            return []

        texts = [self._chat_text(prompt) for prompt in prompts]
        sampling = self.sampling_params_class(
            temperature=0.0,
            max_tokens=1,
            min_tokens=1,
            allowed_token_ids=[self.yes_token_id, self.no_token_id],
            logprob_token_ids=[self.yes_token_id, self.no_token_id],
        )
        results = self.model.generate(texts, sampling_params=sampling, use_tqdm=True)
        if len(results) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(results)} results for {len(prompts)} prompts"
            )

        probabilities: list[YesNoProbability] = []
        for result in results:
            if not result.outputs or not result.outputs[0].logprobs:
                raise RuntimeError("vLLM did not return first-token log probabilities")
            token_logprobs = result.outputs[0].logprobs[0]
            try:
                yes_logprob = self._logprob(token_logprobs[self.yes_token_id])
                no_logprob = self._logprob(token_logprobs[self.no_token_id])
            except KeyError as exc:
                raise RuntimeError(
                    "vLLM did not return both requested yes/no token probabilities"
                ) from exc
            probabilities.append(_normalized_pair(yes_logprob, no_logprob))
        return probabilities

    def score_yes_no(self, prompts: Sequence[str], batch_size: int = 8) -> list[bool]:
        return [
            yes >= no
            for yes, no in self.score_yes_no_probabilities(prompts, batch_size)
        ]
