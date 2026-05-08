# =============================================================================
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2024 Qualcomm Technologies, Inc. All rights reserved.
#
# All data and information contained in or disclosed by this document are
# confidential and proprietary information of Qualcomm Technologies, Inc., and
# all rights therein are expressly reserved. By accepting this material, the
# recipient agrees that this material and the information contained therein
# are held in confidence and in trust and will not be used, copied, reproduced
# in whole or in part, nor its contents revealed in any manner to others
# without the express written permission of Qualcomm Technologies, Inc.
#
# =============================================================================

"""  utility method to evaluate perplexity score on WikiText """
from itertools import chain
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from transformers import default_data_collator
import json
import torch
from PIL import Image
import os
from .vl_utils import get_vl_inputs
from qwen_vl_utils import process_vision_info

class VisualEmbeddingGenerator(torch.nn.Module):

    def __init__(self, visual_patch_embed, visual_blocks, visual_merger, visual_rot_pos_emb, grid_thw, device):
        super().__init__()
        self.patch_embed = visual_patch_embed
        self.blocks = visual_blocks
        self.merger = visual_merger

        self.device = device
        self.rotary_pos_emb_ = visual_rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        self.cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # this forwrad gets the image pixel values that we get from the AutoProcessor when we pass the image and text (text -> input ids, and image-> pixel values)
    def forward(self, hidden_states):
        hidden_states = self.patch_embed(hidden_states.to(self.device))
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens, rotary_pos_emb=self.rotary_pos_emb_)
        return self.merger(hidden_states)


class Qwen2Dataset(Dataset):

    def __init__(self, model, visual, processor, dataset_path, json_file_path, emb_length, calibration, vision_input_size, num_test_batches=300):
        self.model = model
        self.visual = visual
        self.visual.eval()
        self.processor = processor
        self.calibration = calibration
        self.dataset_path = dataset_path
        self.json_file_path = json_file_path
        self.emb_length = emb_length
        self.vision_input_size = vision_input_size
        with open(json_file_path) as file:
            self.json_file = json.load(file)
        self.num_test_batches = num_test_batches

    def __len__(self):
        if self.calibration:
            return len(self.json_file)
        else:
            return self.num_test_batches

    def __getitem__(self, idx):
        json_data = self.json_file[idx] if idx < len(self.json_file) else self.json_file[-1]
        batch = {}
        batch['query'] = json_data
        batch['image_file'] = os.path.join(self.dataset_path, "coco/train2017/000000000025.jpg")
        if self.calibration:
            try:
                batch['image_file'] = os.path.join(self.dataset_path, json_data['image'])
                batch['query'] = json_data['conversations'][2]['value']
            except:
                batch['query'] = "Please create a recipe for me with these ingredients."

        # Append the system prompt first
        # 准备输入消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": batch['image_file'],
                        "resized_height": self.vision_input_size[0],
                        "resized_width": self.vision_input_size[1],
                    },
                    {
                        "type": "text",
                        "text": batch['query']
                    },
                ],
            },
        ]
        # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What feature can be seen on the back of the bus?<|im_end|>\n<|im_start|>assistant\n'
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
                
        # apply qwen2-vl preprocesing, prompt -> token_ids, raw_image -> image_tensor (CLIPImageProcessor)
        inputs = self.processor(text=prompt, images=image_inputs, return_tensors="pt")

        # Extract image features from VEG
        # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
        inputs_embeds = self.model.embed_tokens(inputs['input_ids'].to(self.model.device))
        pixel_values = inputs['pixel_values']
        with torch.no_grad():
            image_embeds = self.visual(pixel_values)
        image_mask = ((inputs['input_ids'] == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device))
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        labels = inputs['input_ids']
        # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py

        emb_length = self.emb_length  # ARN
        #'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is shown in this image?<|im_end|>\n<|im_start|>assistant\n'
        system_start_prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'
        system_start = self.processor.tokenizer(system_start_prompt, return_tensors="pt", add_special_tokens=False)
        system_start_prompt_length = len(system_start["input_ids"][0])  #14

        #'<|im_end|><|im_start|>assistant\n'
        system_end_prompt = '<|im_end|>\n<|im_start|>assistant\n'
        system_end = self.processor.tokenizer(system_end_prompt, return_tensors="pt", add_special_tokens=False)
        system_end_prompt_length = len(system_end["input_ids"][0])  #5

        #'<|vision_start|><|image_pad|>*529<|vision_end|>' 529=image_pad_num
        image_pad_num = (inputs['input_ids'] == torch.tensor(151655)).sum().item()
        image_prompt = f"{'<|vision_start|>'}{'<|image_pad|>' * image_pad_num}{'<|vision_end|>'}"
        image_encoder = self.processor.tokenizer(image_prompt, return_tensors="pt", add_special_tokens=False)
        image_prompt_length = len(image_encoder["input_ids"][0])  #529 +2

        if self.calibration:
            repeated_emb = inputs_embeds
            repeated_labels = labels
            while repeated_emb.shape[1] < emb_length:
                repeated_emb = torch.cat([repeated_emb, inputs_embeds], dim=1)
                repeated_labels = torch.cat([repeated_labels, labels], dim=1)
            inputs_embeds = repeated_emb[:, :emb_length, :].contiguous()
            labels = repeated_labels[:, :emb_length].contiguous()
        else:
            # for ppl, we remove the sys prompt + image embeddings, and use whatever actual wikitext is after that
            cut_embeds = inputs_embeds[:, (system_start_prompt_length + image_prompt_length):, :].contiguous()
            cut_labels = labels[:, (system_start_prompt_length + image_prompt_length):].contiguous()
            if cut_embeds.shape[1] > emb_length:
                # for qwen2 llm, the begiinig 37 is the fixed prompt embedding, the following 576 is the image embedding.
                inputs_embeds = torch.cat([cut_embeds[:, :(emb_length - system_end_prompt_length), :], cut_embeds[:, -system_end_prompt_length:, :]], dim=1)
                labels = torch.cat([cut_labels[:, :(emb_length - system_end_prompt_length)], cut_labels[:, -system_end_prompt_length:]], dim=1)
            else:
                repeated_emb = cut_embeds
                repeated_labels = cut_labels
                while repeated_emb.shape[1] < emb_length:
                    repeated_emb = torch.cat([repeated_emb, cut_embeds], dim=1)
                    repeated_labels = torch.cat([repeated_labels, cut_labels], dim=1)
                inputs_embeds = repeated_emb[:, :emb_length, :].contiguous()
                labels = repeated_labels[:, :emb_length].contiguous()

        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long)

        return {'input_embeddings': inputs_embeds.squeeze(dim=0), 'attention_mask': attention_mask.squeeze(dim=0), 'labels': labels.squeeze(dim=0), 'position_ids': None}
    
    def get_vl_embeds(self, prompt_text, img_path):
        """
        Get the visual embeddings for the given batch.
        """
        
        print("img_path is: ", img_path)
        # Read the raw image and get the image tensor
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "resized_height": self.vision_input_size[0],
                        "resized_width": self.vision_input_size[1],
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                ],
            },
        ]
        # '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What feature can be seen on the back of the bus?<|im_end|>\n<|im_start|>assistant\n'
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
 
        # apply qwen2 preprocesing, prompt -> token_ids, raw_image -> image_tensor (CLIPImageProcessor)
        inputs = self.processor(text=prompt, images=image_inputs, return_tensors="pt")

        # Extract image features from VEG
        # ref:/usr/local/lib/python3.10/dist-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py
        inputs_embeds = self.model.embed_tokens(inputs['input_ids'].to(self.model.device))
        pixel_values = inputs['pixel_values']
        with torch.no_grad():
            image_embeds = self.visual(pixel_values)
        image_mask = ((inputs['input_ids'] == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device))
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds


class CustomDataset(Dataset):
    """
    Dataset for GPTQ-preprocessed tokens
    """

    def __init__(self, tokens, block_size=2048):
        self.full_tokens = tokens
        self.block_size = block_size
        self._len = len(tokens["input_ids"][0]) // block_size

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = (idx + 1) * self.block_size

        input_ids = self.full_tokens["input_ids"][0, start_idx:end_idx]
        labels = input_ids.clone()
        attn_mask = self.full_tokens["attention_mask"][0, start_idx:end_idx]
        output = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
        return output


def _get_column_names(dataset):
    if hasattr(dataset, "column_names"):
        return dataset.column_names
    else:
        return next(iter(dataset.take(1))).keys()


def get_column_name(dataset):
    column_names = _get_column_names(dataset)
    if "text" in column_names:
        return "text"
    else:
        return column_names[0]


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class PreprocessGptqSplit:

    def __init__(self, tokenizer, block_size, add_special_tokens=True):
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._add_special_tokens = add_special_tokens

    def preprocess(self, dataset):
        column_name = get_column_name(dataset)
        tokens = self._tokenizer("\n\n".join(dataset[column_name]), return_tensors="pt", add_special_tokens=self._add_special_tokens)
        dataset = CustomDataset(tokens, self._block_size)
        return dataset


class PreprocessSplit:

    def __init__(self, tokenizer, block_size, column_name="text", add_special_tokens=True):
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._column_name = column_name
        self._add_special_tokens = add_special_tokens

    def _tokenize_fn(self, examples):
        return self._tokenizer(examples[self._column_name], return_token_type_ids=False, add_special_tokens=self._add_special_tokens)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def _group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self._block_size:
            total_length = (total_length // self._block_size) * self._block_size
        # Split by chunks of max_len.
        result = {k: [t[i:i + self._block_size] for i in range(0, total_length, self._block_size)] for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    def preprocess(self, dataset):
        map_kwargs = {
            "num_proc": None,
            "load_from_cache_file": True,
            "desc": "Running tokenizer on dataset",
        }

        tokenized_dataset = dataset.map(
            self._tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
            **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
        )

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
        print(f"[Load Dataset]grouped train dataset")
        map_kwargs["desc"] = f"Grouping texts in chunks of {self._block_size}"
        dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            **(map_kwargs if not isinstance(dataset, IterableDataset) else {}),
        )

        return dataset


def get_wiki_dataset(block_size, tokenizer, cache_dir):
    dataset = {}
    dataset['train'] = load_dataset(path='wikitext', name='wikitext-2-raw-v1', cache_dir=cache_dir, split='train')
    dataset['train'] = PreprocessSplit(tokenizer, block_size).preprocess(dataset['train'])

    dataset['test'] = load_dataset(path='wikitext', name='wikitext-2-raw-v1', cache_dir=cache_dir, split='test')
    dataset['test'] = PreprocessGptqSplit(tokenizer, block_size).preprocess(dataset['test'])

    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=1, collate_fn=default_data_collator)

    return train_dataloader, test_dataloader, dataset


def get_ct_dataset(block_size, tokenizer, cache_dir):
    dataset = {}
    dataset_ct = load_dataset('json', data_files=cache_dir)
    dataset['train'] = PreprocessSplit(tokenizer, block_size).preprocess(dataset_ct['train'])
    dataset['test'] = PreprocessGptqSplit(tokenizer, block_size).preprocess(dataset_ct['train'])

    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=1, collate_fn=default_data_collator)

    return train_dataloader, test_dataloader, dataset


def get_qwen2_dataset(llm_model, qwen2_dataset_setting, num_test_batches):
    # 读取配置
    emb_length = qwen2_dataset_setting['emb_length']
    device = qwen2_dataset_setting['device']
    qwen2vl_model_id = qwen2_dataset_setting['qwen2vl_model_id']
    calibration_dataset_path = qwen2_dataset_setting['calibration_dataset_path']
    ppl_evaluation_dataset_path = qwen2_dataset_setting['ppl_evaluation_dataset_path']
    image_dataset_path = qwen2_dataset_setting['image_dataset_path']
    vision_input_size = qwen2_dataset_setting['vision_input_size']

    #构建model和visual，processor
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    Qwen2VLmodel = Qwen2VLForConditionalGeneration.from_pretrained(qwen2vl_model_id).to(device)
    visual = Qwen2VLmodel.visual
    # llm_model = Qwen2VLmodel.model
    processor = AutoProcessor.from_pretrained(qwen2vl_model_id)

    ## need to change if input image size is different (640, 640)
    grid_thw = torch.tensor([[1, 46, 46]])
    veg = VisualEmbeddingGenerator(visual.patch_embed, visual.blocks, visual.merger, visual.rot_pos_emb, grid_thw, device)

    # 构建数据集
    dataset = {}
    dataset_train = Qwen2Dataset(llm_model, veg, processor, image_dataset_path, calibration_dataset_path, emb_length, calibration=True, vision_input_size=vision_input_size, num_test_batches=num_test_batches)

    dataset_test = Qwen2Dataset(llm_model, veg, processor, image_dataset_path, ppl_evaluation_dataset_path, emb_length, calibration=False, vision_input_size=vision_input_size, num_test_batches=num_test_batches)
    dataset['train'] = dataset_train
    dataset['test'] = dataset_test

    def custom_collate_fn(batch):
        return batch

    train_dataloader = DataLoader(dataset['train'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    test_dataloader = DataLoader(dataset['test'], shuffle=False, batch_size=1, collate_fn=default_data_collator)
    del llm_model, veg, Qwen2VLmodel
    return train_dataloader, test_dataloader, dataset
