# Enhancing Multi-Image Understanding Through Delimiter Token Scaling (ICLR 2026)



by [Minyoung Lee](https://sites.google.com/view/minyoung-lee), [Yeji Park](https://yejipark-m.github.io/), [Dongjun Hwang](https://dongjunhwang.github.io/), [Yejin Kim](https://sites.google.com/view/yejin-c-kim/), [Seong Joon Oh](https://coallaoh.github.io/), [Junsuk Choe](https://sites.google.com/site/junsukchoe/)

This repository contains the code for the paper **["Enhancing Multi-Image Understanding Through Delimiter Token Scaling"](https://openreview.net/forum?id=7QFf05KrOm)** presented at ICLR 2026.


> **Abstract**: Large Vision-Language Models (LVLMs) achieve strong performance on single-image tasks, but their performance declines when multiple images are provided as input. One major reason is the cross-image information leakage, where the model struggles to distinguish information across different images. Existing LVLMs already employ delimiter tokens to mark the start and end of each image, yet our analysis reveals that these tokens fail to effectively block cross-image information leakage. To enhance their effectiveness, we propose a method that scales the hidden states of delimiter tokens. This enhances the model’s ability to preserve image-specific information by reinforcing intra-image interaction and limiting undesired cross-image interactions. Consequently, the model is better able to distinguish between images and reason over them more accurately. Experiments show performance gains on multi-image benchmarks such as Mantis, MuirBench, MIRB and QBench2. We further evaluate our method on text-only tasks that require clear distinction. The method improves performance on multi-document and multi-table understanding benchmarks, including TQABench, MultiNews and WCEP-10. Notably, our method requires no additional training or inference cost.

![](assets/Method_Overview.png)

## TODO / Code Release Plan

We are in the process of cleaning up and preparing the codebase for public release.
The following components will be released progressively:

- [X] **Multi-image understanding evaluation code**  
  (Delimiter token scaling integrated into LVLM inference and evaluation pipelines)

- [ ] **LLM benchmark code**  
  (Multi-document and multi-table benchmarks including TQABench, MultiNews, and WCEP-10)

- [ ] **Visualization code**  
  (Attention maps and interaction analysis for delimiter tokens)

The full code will be released upon final preparation.


## Installation

### Pull the docker image

```bash
docker pull myelena/delim_scaling:slim
```
### Install dependencies

Inside the container:

```bash
git clone https://github.com/MYMY-young/DelimScaling.git
cd DelimScaling

cd transformers
pip install -e .

cd ../qwen-vl-utils
pip install -e .

pip install flash-attn==2.7.4.post1
```

### Running Evaluation

```bash
accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct,device_map=cuda,attn_implementation=flash_attention_2 \
    --tasks mantis \
    --batch_size 1 \
    --delim_scaling True \
    --scale 8 \
    --select_layer 0,1,2,3
```

### Key Arguments

- `--model_args pretrained=...` : Specify the pretrained model to use. This can be either a local path or a HuggingFace model identifier. For example, `Qwen/Qwen2.5-VL-3B-Instruct`.
- `--tasks` : Specify the evaluation tasks. 
- `--delim_scaling` : Enable delimiter token scaling.  
- `--scale` : Scaling factor.
- `--select_layer` : Layers where scaling is applied.

## Supported Tasks

Multi-Image Understanding benchmarks:

- **[Mantis](https://huggingface.co/datasets/TIGER-Lab/Mantis-Eval)**
- **[Muirbench](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH)**
- **[MIRB](https://huggingface.co/datasets/VLLMs/MIRB)**
- **[QBench2](https://huggingface.co/datasets/q-future/Q-Bench2-HF/viewer/default/dev)**
## Supported Models

For Multi-Image Understanding:

- **[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)**
- **[InternVL3](https://huggingface.co/OpenGVLab/InternVL3-1B-hf)**
- **[LLaVA-OneVision](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)**




## Acknowledgments
Our code is based on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [Transformer](https://github.com/huggingface/transformers).
If you use our work, please consider citing the above works as well.

## Citation

If you find this work useful for your research, please consider citing:

```bib
@inproceedings{lee2026delimscale,
  title={Enhancing {Multi-Image} Understanding through {Delimiter Token} Scaling},
  author={Lee, Minyoung and Park, Yeji and Hwang, Dongjun and Kim, Yejin and Oh, Seong Joon and Choe, Junsuk},
  booktitle={Proceedings of the 14th International Conference on Learning Representations},
  year={2026}
}
```
