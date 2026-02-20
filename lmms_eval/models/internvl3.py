import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
)
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.api.instance import Instance
from lmms_eval import utils

@register_model("internvl3")
class InternVL3(lmms):
    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL-Chat-V1-3",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        select_layer: Optional[list[int]] = None,
        delim_scaling: Optional[bool] = None,
        scale: Optional[float] = None,
        task: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map

        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            attn_implementation="flash_attention_2"
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._config = self.model.config

        if accelerator.num_processes > 1:
            self._model = accelerator.prepare(self.model)
            self.accelerator = accelerator
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        self.select_layer = select_layer
        self.delim_scaling = delim_scaling
        self.scale = scale

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        return [j for i in input for j in i]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self.tokenizer.eos_token])
            until = [item for item in until if item != "\n\n"]

            batched_messages = []

            for i, context in enumerate(contexts):
                visuals = []
                for visual in visual_list[i]:
                    if isinstance(visual, Image.Image):
                        buffer = BytesIO()
                        visual.save(buffer, format="JPEG")
                        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_str}"})
                    elif isinstance(visual, str) and visual.startswith("http"):
                        visuals.append({"type": "image", "url": visual})

                content_list = visuals + [{"type": "text", "text": context}]
                message = {"role": "user", "content": content_list}

                batched_messages.append(message)

            inputs = self.processor.apply_chat_template(batched_messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 124),
                    top_p=gen_kwargs.get("top_p", None),
                    num_beams=gen_kwargs.get("num_beams", 1),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=self.use_cache,
                    select_layer=self.select_layer,
                    delim_scaling=self.delim_scaling,
                    scale=self.scale,
                )
                input_len = inputs.input_ids.shape[1]
                generated_ids_trimmed = [out_ids[input_len:] for out_ids in output_ids]
                answers = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

            for i, ans in enumerate(answers):
                for term in until:
                    if term in ans:
                        ans = ans.split(term)[0]
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (contexts[i], gen_kwargs), ans)
                pbar.update(1)

            torch.cuda.empty_cache()

        pbar.close()
        return re_ords.get_original(res)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood not implemented for InternVL3")

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
