import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
import torch.nn.functional as F
import numpy as np
from qwen_vl_utils import process_vision_info


def load_vl_model(model_id, device, config=None):
    if config is None:
        config = AutoConfig.from_pretrained(model_id, cache_dir="cache_dir", trust_remote_code=True)
        setattr(config, '_attn_implementation', 'eager')
    vl_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, config=config).eval().to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return vl_model.visual, vl_model.model, vl_model, processor


def get_vl_inputs(processor, img_path=None, text="", device="cuda:0", img_w=640, img_h=640):
    messages = [{"role": "user", "content": [{"type": "text", "text": text},]}]
    if img_path is not None:
        messages[0]["content"].append({"type": "image", "image": img_path, "resized_height": img_h, "resized_width": img_w})

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    return inputs


def get_vl_embeddings(visual, llm, inputs):
    input_ids = inputs["input_ids"]
    inputs_embeds = llm.embed_tokens(input_ids)

    pixel_values, image_grid_thw = inputs.get("pixel_values", None), inputs.get("image_grid_thw", None)

    if pixel_values is not None:
        pixel_values = pixel_values.type(visual.get_dtype())
        image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == llm.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    return inputs_embeds



def show_pixel_shape_and_grid_thw(model_id, img_shape):
    processor = AutoProcessor.from_pretrained(model_id)
    image_inputs = (np.random.random(img_shape) * 255).astype(np.uint8)
    # Preparation for inference
    inputs = processor(
        text=[""],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    print("pixel_values.shape =", pixel_values.shape)
    print("image_grid_thw =", image_grid_thw)
