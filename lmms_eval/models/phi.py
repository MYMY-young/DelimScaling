import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union
from PIL.Image import Image as ImageObject
from torchvision import transforms

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

@register_model("phi")
class Phi(lmms):
    """
    Phi3 Model
    "https://huggingface.co/microsoft/phi-1_5"
    """

    def __init__(
        self,
        pretrained: str = "microsoft/phi-1_5",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = "flash_attention_2",
        system_prompt: Optional[str] = "You are a helpful assistant.",
        reasoning_prompt: Optional[str] = None,
        select_layer: Optional[list[int]] = None,
        delim_scaling: Optional[bool] = None,
        scale: Optional[float] = None,
        visualize: Optional[bool] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")
        self.select_layer = select_layer
        self.delim_scaling = delim_scaling
        self.scale = scale
        self.visualize = visualize
        self.task = task

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()


        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        if self.task[0] == "multinews" and self.scale != 1.0:
            special_tokens_dict = {'additional_special_tokens': ['|||||']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))

            sep_text = "|||||"
            sep_id = self.tokenizer.convert_tokens_to_ids(sep_text)

        elif ("tqabench_64k" in self.task[0]) and self.scale != 1.0:
            special_tokens_dict = {'additional_special_tokens': ['\n\n## ']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            sep_text = '\n\n## '
            sep_id = self.tokenizer.convert_tokens_to_ids(sep_text)
        elif self.task[0] == "wcep10" and self.scale != 1.0:
            special_tokens_dict = {'additional_special_tokens': ['</s>']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            sep_text = '</s>'
            sep_id = self.tokenizer.convert_tokens_to_ids(sep_text)
        else:
            sep_id = None
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])

            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            batched_messages = []
            for i, context in enumerate(contexts):

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context


                message.append(
                    {
                        "role": "user",
                        "content":  context,
                    }
                )


                batched_messages.append(message)

            texts = [self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)

            if self.scale != 1.0:
                sep_id_index = (inputs["input_ids"] == sep_id).nonzero(as_tuple=False) if sep_id is not None else None
                mask = inputs["input_ids"] == sep_id
                inputs["input_ids"][mask] = self.tokenizer.encode(sep_text, add_special_tokens=False)[0]

            #inputs = [self.tokenizer(msg, return_tensors="pt").to(self.model.device) for msg in batched_messages]
            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            '''offsets = inputs.pop("offset_mapping")
            # Set default generation kwargs
            H2 = re.compile(r'##\s+')
            vs_pos, ve_pos = [], []
            for b_idx, text in enumerate(texts):
                for m in H2.finditer(text):
                    s, e = m.span()  # 문자 위치
                    # 문자 위치 → 토큰 인덱스
                    offs = offsets[b_idx].tolist()
                    tok_s = min(i for i, (st, ed) in enumerate(offs) if st <= s < ed or st >= s)
                    tok_e = min(i for i, (st, ed) in enumerate(offs) if st <= e - 1 < ed or st >= e - 1)
                    vs_pos.append([b_idx, tok_s])
                    ve_pos.append([b_idx, tok_e])'''

            '''for b_idx, (text, offs) in enumerate(zip(texts, offsets)):
                offs = np.asarray(offs.cpu(), dtype=np.int32)
                starts, ends = offs[:, 0], offs[:, 1]

                # ---- "##" 위치 전부 찾기 (C 레벨) ----
                spans = []
                i = 0
                while True:
                    j = text.find("##", i)
                    if j == -1:
                        break
                    spans.append((j, j + 2))  # "##" 길이는 2
                    i = j + 1  # overlap 허용

                if not spans:
                    continue

                s = np.array([st for st, _ in spans], dtype=np.int32)
                e = np.array([ed - 1 for _, ed in spans], dtype=np.int32)  # inclusive

                # ---- 시작 토큰 찾기 ----
                tok_s = np.searchsorted(starts, s, side="right") - 1
                tok_s = np.clip(tok_s, 0, len(starts) - 1)
                bad_s = ~((starts[tok_s] <= s) & (s < ends[tok_s]))
                tok_s[bad_s] = np.clip(tok_s[bad_s] + 1, 0, len(starts) - 1)

                # ---- 끝 토큰 찾기 ----
                tok_e = np.searchsorted(starts, e, side="right") - 1
                tok_e = np.clip(tok_e, 0, len(starts) - 1)
                bad_e = ~((starts[tok_e] <= e) & (e < ends[tok_e]))
                tok_e[bad_e] = np.clip(tok_e[bad_e] + 1, 0, len(starts) - 1)

                vs_pos.extend([[b_idx, int(t)] for t in tok_s])
                ve_pos.extend([[b_idx, int(t)] for t in tok_e])'''

            default_gen_kwargs = {
                "max_new_tokens": 60,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": None,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
            with torch.inference_mode():
                with torch.no_grad():
                    cont = self.model.generate(
                        **inputs,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=pad_token_id,
                        do_sample=current_gen_kwargs["do_sample"],
                        temperature=current_gen_kwargs["temperature"],
                        top_p=current_gen_kwargs["top_p"],
                        num_beams=current_gen_kwargs["num_beams"],
                        max_new_tokens=current_gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        select_layer=self.select_layer,
                        delim_scaling=self.delim_scaling,
                        scale=self.scale,
                        task_name=task,
                        sep_id=sep_id_index
                    )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

            del inputs, cont
            torch.cuda.empty_cache()
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")