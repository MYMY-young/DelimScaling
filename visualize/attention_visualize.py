"""
Qwen2.5-VL attention map visualization + hidden-state (delimiter token) scaling.

- Attention map extraction: standard transformers `output_attentions=True` (no modification needed)
- Hidden-state scaling: added via a monkey-patch on `Qwen2_5_VLModel.forward`
  (paper: "Enhancing Multi-Image Understanding Through Delimiter Token Scaling", ICLR 2026)

Run:
    python3.10 attention_visualize.py \
        --dataset mirb --sample-idx 257 --layer 35 \
        --select-layer 0,1,2,3 --scale 8 --res-patches 512

Required packages (if not already installed in this environment):
    python3.10 -m pip install "transformers>=4.49,<4.51" accelerate qwen-vl-utils[decord] \
        pillow matplotlib seaborn datasets
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")  # headless/server environment, save to file instead of showing a window
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from PIL import Image

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_mod
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

sys.path.insert(0, "/path/to/DelimScaling")  # <-- CHANGE THIS to your local DelimScaling repo path
from lmms_eval.tasks.mantis.utils import mantis_doc_to_text, mantis_doc_to_visual
from lmms_eval.tasks.mirb.utils import mirb_doc_to_text, mirb_doc_to_visual


def comma_separated_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# 0. Config (CLI args, defaults match the values this script has been using)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct", help="local path or HF hub id")
parser.add_argument("--dataset", default="mirb", choices=["mantis", "mirb"])
parser.add_argument("--sample-idx", type=int, default=257, help="dataset row index")
parser.add_argument("--layer", type=int, default=35, help="which layer's attention to visualize (0-indexed)")
parser.add_argument("--select-layer", type=comma_separated_ints, default=[0, 1, 2, 3], help="comma-separated layer indices where delim_scaling is applied, e.g. 0,1,2,3")
parser.add_argument("--scale", type=float, default=8, help="delim_scaling multiplier")
parser.add_argument("--res-patches", type=int, default=512, help="image resolution cap in 28x28 patches (min_pixels=max_pixels=res_patches*28*28)")
parser.add_argument("--out-dir", default="attn_maps")
args = parser.parse_args()

MODEL_ID = args.model_id
DATASET = args.dataset
SAMPLE_IDX = args.sample_idx
LAYER = args.layer
SELECT_LAYERS = args.select_layer
DELIM_SCALE = args.scale
RES_PATCHES = args.res_patches
OUT_DIR = args.out_dir

DATASET_CONFIGS = {
    "mantis": {
        "path": "TIGER-Lab/Mantis-Eval",
        "doc_to_visual": mantis_doc_to_visual,
        "doc_to_text": mantis_doc_to_text,
    },
    "mirb": {
        "path": "VLLMs/MIRB-hf",
        "doc_to_visual": mirb_doc_to_visual,
        "doc_to_text": mirb_doc_to_text,
    },
}

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Hidden-state scaling monkey-patch
#    (when model.model._scaling_config is None, behavior is 100% identical to stock)
# ---------------------------------------------------------------------------
def compute_image_pooled_vecs(hidden_states, vs_pos, ve_pos, pool="mean"):
    assert vs_pos.shape == ve_pos.shape
    vecs = []
    for i in range(vs_pos.shape[0]):
        b, s, e = int(vs_pos[i, 0]), int(vs_pos[i, 1]), int(ve_pos[i, 1])
        tok = hidden_states[b, s : e + 1, :]
        v = tok.mean(dim=0) if tok.numel() else hidden_states.new_zeros(hidden_states.size(-1))
        vecs.append(F.normalize(v, dim=0, eps=1e-9))
    return torch.stack(vecs, dim=0)


def cosine_sim_matrix(img_vecs):
    return img_vecs @ img_vecs.t()


def sim_to_lambda_mean(sim_mat, base=1.0, alpha=1.0, clamp=(0.0, 1.0)):
    N = sim_mat.size(0)
    mask = ~torch.eye(N, dtype=torch.bool, device=sim_mat.device)
    sims = sim_mat[mask].view(N, N - 1)
    if clamp is not None:
        sims = sims.clamp(*clamp)
    return base + alpha * sims.mean(dim=1)


_orig_model_forward = qwen_mod.Qwen2_5_VLModel.forward


def _patched_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    temp_input_ids=None,
    select_layer=None,
    delim_scaling=None,
    scale=None,
):
    scaling_cfg = getattr(self, "_scaling_config", None)
    if scaling_cfg is None:
        return _orig_model_forward(
            self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position, temp_input_ids=temp_input_ids,
            select_layer=select_layer, delim_scaling=delim_scaling, scale=scale,
        )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    temp_input_ids = getattr(self, "_temp_input_ids", None)
    vs_pos = ve_pos = None
    if temp_input_ids is not None:
        vs_pos = (temp_input_ids == self.config.vision_start_token_id).nonzero(as_tuple=False)
        ve_pos = (temp_input_ids == self.config.vision_end_token_id).nonzero(as_tuple=False)

    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0]

        # --- hidden-state scaling hook ---
        if vs_pos is not None and i in scaling_cfg["select_layer"] and len(vs_pos) > 1:
            img_vecs = compute_image_pooled_vecs(hidden_states, vs_pos, ve_pos)
            sim_mat = cosine_sim_matrix(img_vecs)
            lambdas = sim_to_lambda_mean(sim_mat)
            for k in range(vs_pos.size(0)):
                b, vs, ve = int(vs_pos[k, 0]), int(vs_pos[k, 1]), int(ve_pos[k, 1])
                hidden_states[b, vs] *= lambdas[k]
                hidden_states[b, ve] *= lambdas[k]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=next_cache,
        hidden_states=all_hidden_states, attentions=all_self_attns,
    )


qwen_mod.Qwen2_5_VLModel.forward = _patched_model_forward
print("[patch] Qwen2_5_VLModel.forward monkey-patched (hidden-state scaling hook added, disabled by default)")


# ---------------------------------------------------------------------------
# 2. Visualization function
# ---------------------------------------------------------------------------
def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().view(-1)[0].item()
    elif isinstance(x, (list, tuple)):
        x = x[0]
    return int(x)


def visualize_attention_with_tokens_lognorm(
    multihead_attention, vs_pos, ve_pos, output_path="attention_with_segments.png",
    kernel_size=8, stride=8, labels=None,
):
    averaged = torch.mean(multihead_attention, dim=1)[0].float()  # (N, N)
    pooled = torch.nn.functional.avg_pool2d(
        averaged.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=stride
    ).squeeze(0).squeeze(0)

    att = pooled.detach().cpu().numpy()
    N_tokens = averaged.shape[-1]
    H_out = att.shape[0]

    # One segment per image, bounded by its own vs_pos..ve_pos span, plus a
    # trailing text segment covering everything after the last image ends.
    img_starts_tok = sorted({_to_scalar(p[1]) for p in vs_pos})
    img_ends_tok = sorted({_to_scalar(p[1]) for p in ve_pos})
    n_images = len(img_starts_tok)
    if n_images == 0:
        raise ValueError("No segment start positions found in vs_pos.")
    assert len(img_ends_tok) == n_images, "vs_pos/ve_pos count mismatch."

    seg_starts_tok = img_starts_tok + [img_ends_tok[-1] + 1]
    seg_ends_tok = [e + 1 for e in img_ends_tok] + [N_tokens]
    n_seg = n_images + 1

    seg_starts_pool = [s // stride for s in seg_starts_tok]
    seg_ends_pool = [min(H_out - 1, (e - 1) // stride) for e in seg_ends_tok]
    midpoints = [(s + e) / 2.0 for s, e in zip(seg_starts_pool, seg_ends_pool)]

    if labels is None:
        labels = [f"Img{i+1}" for i in range(n_images)] + ["T"]
    else:
        assert len(labels) == n_seg, "len(labels) must match the number of segments."

    # Boundary lines at each image's exact start (vs_pos) and end (ve_pos)
    # token position -- not just a label centered in the segment.
    start_bounds_pool = [s // stride for s in img_starts_tok]
    end_bounds_pool = [min(H_out - 1, e // stride) + 1 for e in img_ends_tok]
    boundary_pool_positions = sorted(set(start_bounds_pool + end_bounds_pool))

    fig, ax = plt.subplots(figsize=(4.0, 4.6), dpi=200)
    norm = LogNorm(vmin=0.0007, vmax=float(att.max()))
    sns.heatmap(
        att, ax=ax, cmap="viridis", norm=norm, square=True, cbar=True,
        cbar_kws={"orientation": "horizontal", "shrink": 0.8, "pad": 0.18},
    )
    ax.set_xticks(midpoints)
    ax.set_yticks(midpoints)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Minor ticks (no labels) mark each image's exact start/end token position,
    # without drawing lines over the attention heatmap itself.
    ax.set_xticks(boundary_pool_positions, minor=True)
    ax.set_yticks(boundary_pool_positions, minor=True)
    ax.tick_params(axis="both", which="minor", length=4, width=0.8, color="black")

    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return pooled


# ---------------------------------------------------------------------------
# 3. Load model / processor
# ---------------------------------------------------------------------------
print(f"[load] loading {MODEL_ID}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",  # required: the only path that actually materializes attention weights
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=RES_PATCHES * 28 * 28, max_pixels=RES_PATCHES * 28 * 28)


# ---------------------------------------------------------------------------
# 4. Prepare input (multi-image sample from the configured dataset)
# ---------------------------------------------------------------------------
ds_cfg = DATASET_CONFIGS[DATASET]

print(f"[data] loading {ds_cfg['path']} ({DATASET})...")
eval_dataset = load_dataset(ds_cfg["path"], split="test")
sample = eval_dataset[SAMPLE_IDX]
print("fields:", list(sample.keys()))

# Use the actual task's doc_to_visual/doc_to_text (same functions + same
# default pre_prompt/post_prompt as tasks/<DATASET>/<DATASET>.yaml) so this
# matches exactly what `--tasks <DATASET>` sends to the model.
pil_images = ds_cfg["doc_to_visual"](sample)  # already converts to RGB
question = ds_cfg["doc_to_text"](sample, lmms_eval_specific_kwargs={"pre_prompt": "", "post_prompt": ""})

print(f"number of images: {len(pil_images)}")
print(f"question: {question}")

messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": img} for img in pil_images]
        + [{"type": "text", "text": question}],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
).to(model.device)

input_ids = inputs["input_ids"]

vision_start_id = model.config.vision_start_token_id
vision_end_id = model.config.vision_end_token_id
vs_pos = (input_ids == vision_start_id).nonzero(as_tuple=False)
ve_pos = (input_ids == vision_end_id).nonzero(as_tuple=False)


# ---------------------------------------------------------------------------
# 5. Baseline forward pass + visualization
# ---------------------------------------------------------------------------
print("[run] baseline forward pass...")
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

print(f"number of layers: {len(outputs.attentions)}, attention shape: {tuple(outputs.attentions[0].shape)}")

baseline_path = os.path.join(OUT_DIR, f"layer_labeled_{DATASET}.png")
visualize_attention_with_tokens_lognorm(
    outputs.attentions[LAYER].float().cpu(), vs_pos, ve_pos,
    output_path=baseline_path,
)
print(f"[saved] {baseline_path}")

del outputs
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 6. Hidden-state scaling forward pass + visualization
# ---------------------------------------------------------------------------
print("[run] hidden-state scaling forward pass...")
model.model._temp_input_ids = input_ids
model.model._scaling_config = {"select_layer": SELECT_LAYERS}

with torch.no_grad():
    outputs_scaled = model(**inputs, output_attentions=True)

model.model._scaling_config = None  # turn it back off

scaled_path = os.path.join(OUT_DIR, f"layer_labeled_scaled_{DATASET}.png")
visualize_attention_with_tokens_lognorm(
    outputs_scaled.attentions[LAYER].float().cpu(), vs_pos, ve_pos,
    output_path=scaled_path,
)
print(f"[saved] {scaled_path}")

del outputs_scaled
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 7. Baseline vs Scaled comparison image
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(Image.open(baseline_path))
axes[0].set_title("Baseline")
axes[0].axis("off")
axes[1].imshow(Image.open(scaled_path))
axes[1].set_title(f"Hidden-state scaled (layers {SELECT_LAYERS})")
axes[1].axis("off")
plt.tight_layout()
compare_path = os.path.join(OUT_DIR, f"comparison_{DATASET}.png")
plt.savefig(compare_path, bbox_inches="tight")
print(f"[saved] {compare_path}")


# ---------------------------------------------------------------------------
# 8. Real delim_scaling forward pass (the repo's actual select_layer/
#    delim_scaling/scale flat-multiply mechanism, same as `--delim_scaling
#    True --scale ... --select_layer ...` in the eval CLI) + visualization
# ---------------------------------------------------------------------------

# Control pass: same call path (select_layer/scale passed through), but with
# delim_scaling explicitly off -- an apples-to-apples ablation of the on/off
# switch itself, rather than reusing the earlier (differently-called) baseline.
print(f"[run] delim_scaling OFF control pass (real mechanism, layers={SELECT_LAYERS}, scale={DELIM_SCALE})...")
with torch.no_grad():
    outputs_delim_off = model(
        **inputs, output_attentions=True,
        select_layer=SELECT_LAYERS, delim_scaling=False, scale=DELIM_SCALE,
    )

delim_off_path = os.path.join(OUT_DIR, f"layer_labeled_delim_scaling_off_{DATASET}_{DELIM_SCALE}.png")
visualize_attention_with_tokens_lognorm(
    outputs_delim_off.attentions[LAYER].float().cpu(), vs_pos, ve_pos,
    output_path=delim_off_path,
)
print(f"[saved] {delim_off_path}")

del outputs_delim_off
torch.cuda.empty_cache()

print(f"[run] delim_scaling ON pass (real mechanism, layers={SELECT_LAYERS}, scale={DELIM_SCALE})...")
with torch.no_grad():
    outputs_delim_scaled = model(
        **inputs, output_attentions=True,
        select_layer=SELECT_LAYERS, delim_scaling=True, scale=DELIM_SCALE,
    )

delim_scaled_path = os.path.join(OUT_DIR, f"layer_labeled_delim_scaled_{DATASET}_{DELIM_SCALE}.png")
visualize_attention_with_tokens_lognorm(
    outputs_delim_scaled.attentions[LAYER].float().cpu(), vs_pos, ve_pos,
    output_path=delim_scaled_path,
)
print(f"[saved] {delim_scaled_path}")

del outputs_delim_scaled
torch.cuda.empty_cache()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(Image.open(delim_off_path))
axes[0].set_title(f"delim_scaling OFF (layers {SELECT_LAYERS})")
axes[0].axis("off")
axes[1].imshow(Image.open(delim_scaled_path))
axes[1].set_title(f"delim_scaling ON (layers {SELECT_LAYERS}, scale={DELIM_SCALE})")
axes[1].axis("off")
plt.tight_layout()
delim_compare_path = os.path.join(OUT_DIR, f"comparison_delim_scaling_on_vs_off_{DATASET}_{DELIM_SCALE}.png")
plt.savefig(delim_compare_path, bbox_inches="tight")
print(f"[saved] {delim_compare_path}")

print("\nDone! Results saved to:", os.path.abspath(OUT_DIR))
