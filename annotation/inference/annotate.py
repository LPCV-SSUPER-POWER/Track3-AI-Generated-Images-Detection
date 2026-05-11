"""
Generate annotations from an image-list JSON (batched).

Step 1 (annotate) main script of the annotation pipeline.
For each image, Qwen2.5-VL is asked 4 times:
  a_step1: image + prompt x 3 (A_STEP1_PROMPTS + FAKE_HINT, hardcoded in this file)
  a_step2: text-only x 1, per_criterion JSON synthesis (prompts/a_step2.txt)

Output: one raw annotation JSON per image.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

TOKEN_LIMIT = 1024

A_STEP1_PROMPTS = [
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following three aspects: Edge & Boundary Integrity, Texture & Resolution Coherence, Material & Object Detail Fidelity.",
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following three aspects: Physical & Common Sense Logic, Text & Symbol Authenticity, Human & Biological Structure Integrity.",
    "Is this image real or fake? Think step-by-step before giving a conclusion. Please analyze based on the following two aspects: Lighting & Shadow Consistency, Perspective & Spatial Accuracy.",
]

FAKE_HINT = "Note: This image is known to be AI-generated. Please identify specific AIGC artifacts in each criterion.\n\n"

# Self-contained prompt path: prompts/a_step2.txt next to the annotation/ folder
_A_STEP2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "a_step2.txt")
A_STEP2_PROMPT = open(_A_STEP2_PATH).read().strip()


def load_model(model_path, device="cuda"):
    print(f"Loading model: {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def batch_vlm(model, processor, img_paths, prompt, max_new_tokens=TOKEN_LIMIT):
    """Batch inference with images."""
    all_texts = []
    all_image_inputs = []
    valid = []

    for i, img_path in enumerate(img_paths):
        try:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            all_texts.append(text)
            all_image_inputs.extend(image_inputs)
            valid.append(i)
        except Exception as e:
            pass

    if not all_texts:
        return {}

    inputs = processor(
        text=all_texts,
        images=all_image_inputs if all_image_inputs else None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    results = {}
    for batch_idx, orig_idx in enumerate(valid):
        generated = output_ids[batch_idx][inputs.input_ids.shape[1]:]
        text = processor.decode(generated, skip_special_tokens=True)
        results[orig_idx] = text

    return results


def batch_text_only(model, processor, prompts, max_new_tokens=TOKEN_LIMIT):
    """Batch inference text-only (no images)."""
    all_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)

    inputs = processor(
        text=all_texts,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    results = []
    for batch_idx in range(len(prompts)):
        generated = output_ids[batch_idx][inputs.input_ids.shape[1]:]
        text = processor.decode(generated, skip_special_tokens=True)
        results.append(text)

    return results


def parse_json_text(response):
    for pattern in [r'\{[\s\S]*\}', r'```json\s*([\s\S]*?)```']:
        match = re.search(pattern, response)
        if match:
            try:
                text = match.group(1) if '```' in pattern else match.group()
                return json.loads(text)
            except:
                pass
    return None


def make_image_id(source, image_path):
    return f"{source}_{Path(image_path).stem}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--list_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model, processor = load_model(args.model_path, args.device)

    with open(args.list_json) as f:
        all_images = json.load(f)

    # Filter already done
    todo = []
    skip = 0
    for row in all_images:
        image_id = make_image_id(row["source"], row["image_path"])
        out_path = Path(args.output_dir) / f"{image_id}.json"
        if out_path.exists():
            skip += 1
        else:
            todo.append(row)

    total = len(todo)
    print(f"Total: {len(all_images)}, Skip: {skip}, Todo: {total}, Batch: {args.batch_size}")

    done = 0
    ok = 0
    fail = 0
    t0 = time.time()

    for bi in range(0, total, args.batch_size):
        batch = todo[bi:bi + args.batch_size]
        bs = len(batch)
        bt0 = time.time()

        # Stage 1: 3 prompts, each batched
        # a_step1_responses[i] = [resp1, resp2, resp3] for batch item i
        a_step1_responses = [[] for _ in range(bs)]

        for prompt_idx, base_prompt in enumerate(A_STEP1_PROMPTS):
            img_paths = []
            prompts = []
            for row in batch:
                is_fake = row["label"] in ("ai-generated", "fake")
                prompt = (FAKE_HINT + base_prompt) if is_fake else base_prompt
                img_paths.append(row["image_path"])
                prompts.append(prompt)

            try:
                results = batch_vlm(model, processor, img_paths, None, max_new_tokens=TOKEN_LIMIT)
                # batch_vlm uses single prompt - need per-image prompts
                # Rewrite: process each with its own prompt in batch
                all_texts = []
                all_image_inputs = []
                valid = []
                for i, (ip, p) in enumerate(zip(img_paths, prompts)):
                    try:
                        messages = [{"role": "user", "content": [
                            {"type": "image", "image": f"file://{ip}"},
                            {"type": "text", "text": p},
                        ]}]
                        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, _ = process_vision_info(messages)
                        all_texts.append(text)
                        all_image_inputs.extend(image_inputs)
                        valid.append(i)
                    except Exception as e:
                        pass

                if all_texts:
                    inputs = processor(
                        text=all_texts,
                        images=all_image_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to(model.device)

                    with torch.inference_mode():
                        output_ids = model.generate(**inputs, max_new_tokens=TOKEN_LIMIT, do_sample=False)

                    for batch_idx, orig_idx in enumerate(valid):
                        generated = output_ids[batch_idx][inputs.input_ids.shape[1]:]
                        resp = processor.decode(generated, skip_special_tokens=True)
                        a_step1_responses[orig_idx].append(resp)

            except Exception as e:
                print(f"  [ERROR] prompt {prompt_idx} batch failed: {e}")
                # Fill empty responses
                for i in range(bs):
                    if len(a_step1_responses[i]) <= prompt_idx:
                        a_step1_responses[i].append("")

        # Stage 2: text-only batch
        a_step2_inputs = []
        for i in range(bs):
            combined = "\n\n".join([f"[Analysis {j+1}]\n{r}" for j, r in enumerate(a_step1_responses[i])])
            a_step2_inputs.append(f"{combined}\n\n{A_STEP2_PROMPT}")

        try:
            a_step2_responses = batch_text_only(model, processor, a_step2_inputs)
        except Exception as e:
            print(f"  [ERROR] a_step2 batch failed: {e}")
            a_step2_responses = [""] * bs

        # Save results
        for i in range(bs):
            row = batch[i]
            image_id = make_image_id(row["source"], row["image_path"])
            is_fake = row["label"] in ("ai-generated", "fake")

            result = parse_json_text(a_step2_responses[i])
            if result:
                ok += 1
                if is_fake:
                    result["overall_likelihood"] = "AI-Generated"
                else:
                    result["overall_likelihood"] = "Real"
                    if "per_criterion" in result:
                        for c in result["per_criterion"]:
                            c["aigc score"] = 0
            else:
                fail += 1
                continue

            result["_meta"] = {
                "image_id": image_id,
                "image_path": row["image_path"],
                "label": row["label"],
                "source": row["source"],
                "generator": row["generator"],
                "a_step1_responses": a_step1_responses[i],
                "elapsed_sec": round((time.time() - bt0) / bs, 1),
                "annotation_model": "Qwen2.5-VL-7B-Instruct",
            }

            out_path = Path(args.output_dir) / f"{image_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        done += bs
        elapsed = time.time() - t0
        rate = done / elapsed * 60 if elapsed > 0 else 0
        eta = (total - done) / (rate / 60) / 3600 if rate > 0 else 0
        batch_time = time.time() - bt0
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{total} | ok={ok} fail={fail} | {rate:.1f}/min | batch {batch_time:.1f}s | ETA {eta:.1f}h")

    elapsed = time.time() - t0
    print(f"DONE: {done}/{total} | ok={ok} fail={fail} | {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
