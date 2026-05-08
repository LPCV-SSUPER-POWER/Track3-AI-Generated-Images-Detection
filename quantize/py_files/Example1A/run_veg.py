#!/usr/bin/env python
# coding: utf-8

# # VEG QuantSim recipe
# Following notebook illustrates the VEG Quantsim which is used as input for the Example_1B
# 
# Illustrates the following-
# 
# - Model building
# - Model Adaptation
# - Dataset Processing using HF APIs
# - QSim creation
# - Adaround
# - Model Preparation
# - Calibration
# - Export
# - SQNR analysis

# In[ ]:


# Install packages only if running in jupyter notebook mode
if hasattr(__builtins__,'__IPYTHON__'):
    get_ipython().system('sudo -H python3 -m pip install --upgrade pip')
    # -------------------------------------------------------------------------------------------------
    # HuggingFace libraries
    # -------------------------------------------------------------------------------------------------
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir tokenizers==0.20')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir transformers==4.45.0')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir accelerate==0.21.0')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir jinja2==3.1.0')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir qwen-vl-utils')
    # -------------------------------------------------------------------------------------------------
    # Other libraries
    # -------------------------------------------------------------------------------------------------
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir sentencepiece')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir pynvml')
    get_ipython().system('sudo -H pip install --quiet --upgrade --root-user-action=ignore --no-cache-dir deepspeed==0.13')




import torch
import torch.nn.functional as F
import torch.nn as nn

_pos_emb, _attention_mask = None, None
torch_dtype = torch.float32

class VisualEmbeddingGenerator(torch.nn.Module):

    def __init__(self, visual, grid_thw, pixel_values_shape_0, device, torch_dtype_val):
        super().__init__()
        from veg_utils.qc_adaptation import Conv2dInplaceConv3d

        self.patch_embed = visual.patch_embed
        self.patch_embed.proj = Conv2dInplaceConv3d(visual.patch_embed.proj)
        self.blocks = visual.blocks
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        self.cu_seqlens = cu_seqlens
        self.projector = visual.merger

        global _pos_emb, _attention_mask
        pos_emb = visual.rot_pos_emb(grid_thw)
        cos, sin = pos_emb.cos(), pos_emb.sin()

        _pos_emb = (cos, sin)

        seq_length = pixel_values_shape_0
        attention_mask = torch.full([1, seq_length, seq_length], torch.finfo(torch_dtype_val).min, device=device, dtype=torch_dtype_val)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i], cu_seqlens[i - 1]:cu_seqlens[i]] = 0
        _attention_mask = attention_mask.unsqueeze(0)

    def forward(self, pixel_values, cos, sin, mask):
        hidden_states = self.patch_embed(pixel_values)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=self.cu_seqlens, rotary_pos_emb=(cos,sin), attention_mask=mask)

        image_features = self.projector(hidden_states)
        return image_features


def main():
    # ## Setting QNN SDK
    # ***
    # Setting up model preparer pro to use QNN 2.31 SDK

    # In[ ]:


    import sys
    import os
    import copy
    import torch
    import numpy as np
    from aimet_torch.pro.utils.profiler import event_marker
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel,VisionMlp, VisionAttention, PatchMerger

    QNN_SDK_ROOT='/opt/qcom/aistack/qairt/2.31.0.250130' # qnn2.31 
    assert QNN_SDK_ROOT != '', 'Please point the QNN_SDK_ROOT variable to your QNN SDK'
    sys.path.insert(0, QNN_SDK_ROOT + '/lib/python')
    os.environ['LD_LIBRARY_PATH'] = os.path.join(QNN_SDK_ROOT + '/lib/x86_64-linux-clang', os.getenv('LD_LIBRARY_PATH', ''))

    os.environ['HF_HOME'] = os.getenv('HF_HOME', '../tmp/hf_home')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    torch.cuda.empty_cache()


    # # Loading config

    # In[ ]:


    import json
    import argparse
    from argparse import Namespace

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, help='Experiment name (e.g. FINAL_RESULTS)')
    args = parser.parse_args()

    # Self-contained ROOT detection: env BUNDLE_ROOT (set by wrapper) or
    # auto-derive from script location (../../.. of this file's dir).
    BUNDLE_ROOT = os.environ.get('BUNDLE_ROOT') or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../..'))

    with open('config/veg_config.json', 'rt') as f:
        config = Namespace(**json.load(f))

    # Override output_dir from exp-name
    config.output_dir = f'{BUNDLE_ROOT}/results/{args.exp_name}/Example1A'
    os.makedirs(config.output_dir, exist_ok=True)
    print(f'[run_veg] exp_name={args.exp_name}')
    print(f'[run_veg] output_dir={config.output_dir}')


    # In[ ]:


    sys.path.append('.')
    from utilities.nsptargets import NspTargets

    # Android GEN2/GEN4 is supported for this notebook. Use GEN4 for Pakala based device, GEN2 for Kailua based device. 
    nsp_target = NspTargets.Android.GEN4

    # Select quantsim config based on target — auto-detect aimet_common install location
    import aimet_common
    aimet_common_dir = os.path.dirname(aimet_common.__file__)
    htp_config_file = f'{aimet_common_dir}/quantsim_config/htp_quantsim_config_{nsp_target.dsp_arch}.json'


    # # Loading model

    # In[ ]:


    from transformers import AutoConfig, Qwen2VLForConditionalGeneration, AutoProcessor
    # can add kwargs here
    kwargs = {}
    kwargs['device_map'] = "auto"
    kwargs['torch_dtype'] = torch.float16  if config.half_precision else torch.float32
    kwargs['cache_dir'] = config.cache_dir
    device = "cuda:0"

    model_id = f"{BUNDLE_ROOT}/models/{args.exp_name}_merged_stage2"
    print(f"[run_veg] model_id={model_id}")

    vl_config = AutoConfig.from_pretrained(model_id, cache_dir=config.cache_dir, trust_remote_code=True)
    setattr(vl_config, '_attn_implementation', 'eager')

    debug_mode = False
    if debug_mode:
        setattr(vl_config.vision_config, "depth", 2)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, config=vl_config).eval().to(device)
    model = model.visual
    processor = AutoProcessor.from_pretrained(model_id, **kwargs)


    # In[ ]:


    from veg_utils.utils import load_and_preprocess_images, get_dummy_input, to_device
    img_h, img_w = config.inp_img_h, config.inp_img_w
    pixel_values, grid_thw = get_dummy_input(processor, img_h, img_w)
    print("pixel_values shape:",pixel_values.shape, ", grid_thw value:",grid_thw)
    ori_output = model(pixel_values.cuda(), grid_thw.cuda() )
    print("output shape:", ori_output.shape)


    # ### Adaround on VEG(~ 7h)

    # In[ ]:


    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    from aimet_torch.pro.quantsim import QuantizationSimModel

    def apply_adaround_veg(model, adaround_data, config, adaround_dir):
        os.makedirs(adaround_dir, exist_ok=True)
        filename_prefix = 'parameters_mha'

        def _dummy_fw(model, inputs):
            if isinstance(inputs, (list, tuple)):
                inputs=[inp.to(device) for inp in inputs]
                return model(*inputs)
            else:
                return model(inputs.to(device))

        dummy_input = adaround_data[0]
        dummy_input = tuple(inp.to(device) for inp in dummy_input) if isinstance(dummy_input, (list, tuple)) else dummy_input.to(device)

        params = AdaroundParameters(data_loader=adaround_data, num_batches=len(adaround_data), default_num_iterations=config.adaround_iter, forward_fn=_dummy_fw)

        quant_sim = QuantizationSimModel(model, dummy_input=dummy_input, quant_scheme=config.quant_scheme, default_param_bw=config.parameter_bit_width, config_file=htp_config_file)

        # Enable PCQ for MHA linear layers({qkv}_proj) only so that when replaced with conv2d layers, encodings data can be split.
        for name, module in quant_sim.model.named_modules():
            if isinstance(module, VisionAttention):
                for n, m in module.named_modules():
                    if isinstance(m, QcQuantizeWrapper) and isinstance(m._module_to_wrap, torch.nn.Linear):
                        m.enable_per_channel_quantization()

        # Disable input/output activations quantizers and compute parameter encodings
        Adaround._compute_param_encodings(quant_sim)

        ada_model = Adaround._apply_adaround(quant_sim, model, dummy_input, params, path=adaround_dir, filename_prefix=filename_prefix)
        # save
        torch.save(ada_model.state_dict(), os.path.join(adaround_dir, "state_dict_mha.pt"))
        encoding_path = os.path.join(adaround_dir, f"{filename_prefix}.encodings")
        return ada_model, encoding_path


    def load_adaround_path_veg(model, path):
        print(f'Load AdaRound path for Visual Embedding Generator from {path}')
        model.load_state_dict(torch.load(os.path.join(path, "state_dict_mha.pt")))
        encoding_path = os.path.join(path, "parameters_mha.encodings")
        return model, encoding_path

    encoding_path = None

    if config.apply_adaround:
        print("apply adaround....")
        if os.path.exists(os.path.join(config.load_adaround_path, "state_dict_mha.pt")):
            model, encoding_path = load_adaround_path_veg(model, config.load_adaround_path)
        else:
            adaround_dir = os.path.join(config.output_dir, 'adaround_veg')
            preprocessed_images = load_and_preprocess_images(config.calibration_images, config.adaround_samples, processor, img_h, img_w)
            data = [(img, grid_thw) for img in preprocessed_images]
            model, encoding_path = apply_adaround_veg(model, data, config, adaround_dir)


    # In[ ]:


    from veg_utils.calc import Calculator
    def compare(data1, data2):
        data1 = data1.detach().cpu().numpy()
        data2 = data2.detach().cpu().numpy()
        print("cosineSim:",Calculator.cosineSim(data1, data2))
        print("sqnr:",Calculator.sqnr(data1, data2))

    out = model(pixel_values.cuda(), grid_thw.cuda())
    compare(ori_output, out)


    # ## Model Adaptations

    # In[ ]:


    from veg_utils.qc_adaptation import replace_visual_attention_with_adaptation, Conv2dInplaceConv3d, replace_linears_with_convs

    model_adapted = replace_visual_attention_with_adaptation(copy.deepcopy(model))
    if config.linear_to_conv:
        model_adapted = replace_linears_with_convs(model_adapted)
    out = model_adapted(pixel_values.cuda(), grid_thw.cuda())
    compare(ori_output, out)


    # ### Merge vision tower + projector

    # In[ ]:


    # VisualEmbeddingGenerator는 모듈 최상위에 정의됨 (pickle 호환)


    veg = VisualEmbeddingGenerator(model_adapted, grid_thw.to(device), pixel_values.shape[0], device, torch_dtype).to(device)
    out = veg(pixel_values.to(device),_pos_emb[0],_pos_emb[1], _attention_mask)
    compare(ori_output, out)


    # #### Clear Model

    # In[ ]:


    del model
    del model_adapted


    # ### Prepare model (11min ~ 12min)

    # In[ ]:


    from aimet_torch.utils import load_pytorch_model
    from aimet_torch.pro.model_preparer import prepare_model as prepare_model_pro

    input_names=['pixel_values', 'position_ids_cos', 'position_ids_sin', 'mask']
    output_names=["vision_embedding"]
    def _prepare_model(model):
        filename = 'visual_embedding_generator'
        model_name = 'VisualEmbeddingGenerator'
        prepare_dir = os.path.join(config.output_dir, "veg_prepared")
        os.makedirs(prepare_dir, exist_ok=True)
        if os.path.exists(os.path.join(prepare_dir, "visual_embedding_generator.py")) and hasattr(__builtins__, '__IPYTHON__'):
            print("Reloading model from Model Preparer Pro")
            model = load_pytorch_model(path=prepare_dir, filename=filename, model_name=model_name, load_state_dict=True).to('cuda')
        else:
            print("Converting model with Model Preparer Pro")
            dummy_input = (pixel_values.to(device), _pos_emb[0],_pos_emb[1], _attention_mask)

            converter_args = []
            for name in input_names:
                converter_args += ['--input_layout', name,'NONTRIVIAL']

            model = prepare_model_pro(model,
                                      dummy_input,
                                      prepare_dir,
                                      filename=filename,
                                      model_name=model_name,
                                      input_names=input_names,
                                      output_names=output_names,
                                      converter_args=converter_args,
                                      onnx_export_args={'opset_version': 17})
        return model


    prepared_model = copy.deepcopy(veg)
    if config.use_model_preparer_pro:
        prepared_model = _prepare_model(prepared_model)
        out = prepared_model(pixel_values.cuda(), _pos_emb[0], _pos_emb[1], _attention_mask)
        compare(ori_output, out)


    # ### Computing the encodings

    # In[ ]:


    from veg_utils.utils import load_and_preprocess_images, get_dummy_input, to_device

    from aimet_common.defs import QuantizationDataType
    from aimet_torch.quantsim import load_encodings_to_sim
    from veg_utils.aimet_quantsim import layernorm_exceptions, matmul_exceptions, enable_concat_input_quantizers, buffer_concat_exceptions, compute_sqnr, get_sqnr

    from tqdm import tqdm

    def _forward_pass_calibration_samples(model, preprocessed_images):
        for preprocessed_image in tqdm(preprocessed_images):
            with torch.no_grad():
                preprocessed_image = preprocessed_image.to(device)
                _ = model(preprocessed_image, _pos_emb[0], _pos_emb[1], _attention_mask)


    def get_quantsim(model):
        dummy_input = (pixel_values.to(device), _pos_emb[0], _pos_emb[1], _attention_mask)
        quant_sim = QuantizationSimModel(
            model=model,
            quant_scheme=config.quant_scheme,
            dummy_input=dummy_input,
            default_param_bw=8,
            default_output_bw=16,
            in_place=config.in_place,
            config_file=htp_config_file)
        # mix precision
        from veg_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig
        quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file="config/mix_precision_w8a16.json")
        quantsim_adjuster.apply_exceptions(quant_sim)

        # Pre-calibration exceptions
        enable_concat_input_quantizers(quant_sim)
        if config.use_16_8_matmul:
            matmul_exceptions(quant_sim)

        if config.load_encodings_path:
            print("Loading pre-trained encodings for VisualEmbeddingGenerator")
            load_encodings_to_sim(quant_sim, config.load_encodings_path)
        else:
            print("Computing encodings for VisualEmbeddingGenerator")
            preprocessed_images = load_and_preprocess_images(config.calibration_images, config.num_calibration_samples, processor, img_h, img_w)
            quant_sim.compute_encodings(_forward_pass_calibration_samples, preprocessed_images)

        # Post-calibration exceptions
        buffer_concat_exceptions(quant_sim)  # overwrite encodings to guarantee match

        return quant_sim


    veg_sim = get_quantsim(prepared_model)

    with torch.no_grad():
        out = veg_sim.model(pixel_values.cuda(), _pos_emb[0], _pos_emb[1], _attention_mask)
        compare(ori_output, out)


    # In[ ]:


    from aimet_torch.onnx_utils import OnnxExportApiArgs
    from aimet_torch.utils import change_tensor_device_placement


    def export_quantsim_model(qsim, output_path, dummy_input, filename_prefix, verbose=False, opset_version=16, input_names=None, output_names=None):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # onnx_api_args = {"opset_version": opset_version, "input_names": input_names, "output_names": output_names, "verbose": verbose}
        onnx_api_args = OnnxExportApiArgs(opset_version=opset_version,
                                          input_names=input_names, output_names=output_names )

        dummy_input = change_tensor_device_placement(dummy_input, torch.device('cpu'))
        device = "cuda"
        qsim.model.cpu()
        qsim.export(path=output_path, filename_prefix=filename_prefix, dummy_input=dummy_input, onnx_export_args=onnx_api_args)
        print(f"ONNX saved at {output_path}")
        qsim.model.to(device)


    # # Export (~9min)

    # In[ ]:


    export_dir = os.path.join(config.output_dir, "veg_exports")
    export_quantsim_model(veg_sim,
                          export_dir,
                          (pixel_values, _pos_emb[0], _pos_emb[1], _attention_mask),
                          "veg",
                          device,
                          input_names=input_names,
                          output_names=output_names)
    pixel_values.cpu().detach().numpy().astype(np.float32).tofile(os.path.join(export_dir, "preprocessed_image.raw"))
    _pos_emb[0].cpu().detach().numpy().astype(np.float32).tofile(os.path.join(export_dir, "position_ids_cos.raw"))
    _pos_emb[1].cpu().detach().numpy().astype(np.float32).tofile(os.path.join(export_dir, "position_ids_sin.raw"))
    _attention_mask.cpu().detach().numpy().astype(np.float32).tofile(os.path.join(export_dir, "mask.raw"))


    # In[ ]:


    with torch.no_grad():
        out = veg_sim.model(pixel_values.cuda(), _pos_emb[0], _pos_emb[1], _attention_mask)
        compare(ori_output, out)


    # ### Computing the SQNR (1min ~ 2min)

    # In[ ]:


    from veg_utils.utils import load_and_preprocess_images
    veg.to(device)
    veg_sim.model.to(device)
    veg.eval()
    veg_sim.model.eval()
    preprocessed_images = load_and_preprocess_images(config.calibration_images, config.num_sqnr_samples,processor,img_h,img_w)
    preprocessed_images=[t.to(device) for t  in preprocessed_images]

    sqnr = compute_sqnr(preprocessed_images, veg, veg_sim.model,_pos_emb,_attention_mask)

    print("FP-Sim SQNR:", float(sqnr))


    # Copyright (c) 2024 Qualcomm Technologies, Inc. and/or its subsidiaries.

    # In[ ]:







if __name__ == "__main__":
    main()
