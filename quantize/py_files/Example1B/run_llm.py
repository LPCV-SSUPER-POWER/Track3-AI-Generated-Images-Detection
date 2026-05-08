#!/usr/bin/env python
# coding: utf-8

# In[ ]:




def main():
    import sys, os, pathlib
    import yaml


    # In[ ]:


    sys.path.append('.')


    # In[ ]:


    # Read the notebook config file
    from aimet_utils.DotDict import DotDict, custom_nb_config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, help='Experiment name')
    args = parser.parse_args()

    # Self-contained ROOT detection: env BUNDLE_ROOT (set by wrapper) or
    # auto-derive from script location (../../.. of this file's dir).
    BUNDLE_ROOT = os.environ.get('BUNDLE_ROOT') or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../..'))

    with open('config/nb_config_tang.yml', 'r') as f:
        nb_cfg = yaml.safe_load(f)

    # Override from --exp-name
    nb_cfg['model']['model_id'] = f'{BUNDLE_ROOT}/models/{args.exp_name}_merged_stage2'
    nb_cfg['output_dir'] = f'{BUNDLE_ROOT}/results/{args.exp_name}/Example1B'
    # Override calib JSON path to the bundle-local copy (auto-tracks BUNDLE_ROOT
    # so the bundle stays portable when moved to a new location).
    _calib_json = f'{BUNDLE_ROOT}/quantize/py_files/local_data/llava_v1_5_mix665k.json'
    nb_cfg['calib']['calibration_dataset_path'] = _calib_json
    nb_cfg['calib']['ppl_evaluation_dataset_path'] = _calib_json

    # Resolve <aimet_common_dir> placeholder in htp_config_file to the active install.
    if '<aimet_common_dir>' in nb_cfg['model'].get('htp_config_file', ''):
        import aimet_common
        _aimet_common_dir = os.path.dirname(aimet_common.__file__)
        nb_cfg['model']['htp_config_file'] = nb_cfg['model']['htp_config_file'].replace(
            '<aimet_common_dir>', _aimet_common_dir)

    nb_cfg = DotDict.from_dict(custom_nb_config(nb_cfg))
    print(f'[run_llm] exp_name={args.exp_name}')
    print(f'[run_llm] model_id={nb_cfg.model.model_id}')
    print(f'[run_llm] output_dir={nb_cfg.output_dir}')


    # In[ ]:


    # Setting NSP Target
    # Select quantsim config based on target
    htp_config_file = nb_cfg.model.htp_config_file
    device = 'cuda'
    ARN = nb_cfg.model.ARN 


    # In[ ]:


    from utilities.profiler import event_marker

    from huggingface.baseline_models.qwen2 import modeling_qwen2 as modeling_qwen2
    from transformers import cache_utils
    from aimet_torch.pro.utils.profiler import event_marker
    from llm_utils.qc_adaptation import (QcAttention, bypass_update_causal_mask, MLP_prepare_conv, ForCausalLM_prepare_conv, MLP_forward_conv, DynamicCache_update, DynamicCache_get_seq_length,
                                         update_attr)


    # In[ ]:


    with event_marker("FP model adaptation configuration"):
        modeling_qwen2.QWEN2_ATTENTION_CLASSES['eager'] = QcAttention

        # Bypass attention_mask preparation
        assert update_attr(modeling_qwen2.Qwen2Model, '_update_causal_mask', bypass_update_causal_mask) or \
            update_attr(modeling_qwen2.Qwen2Model, '_prepare_decoder_attention_mask', bypass_update_causal_mask),  \
                f"neither _prepare_decoder_attention_mask(..) nor _update_causal_mask(..) found, Unknown Qwen2Model definition in {modeling_qwen2.__file__}"

        # Adaptation to use Conv instead of linear
        setattr(modeling_qwen2.Qwen2MLP, 'prepare_conv', MLP_prepare_conv)
        setattr(modeling_qwen2.Qwen2MLP, 'forward_conv', MLP_forward_conv)
        setattr(modeling_qwen2.Qwen2ForCausalLM, 'prepare_conv', ForCausalLM_prepare_conv)

        # Adapting KV$ management
        assert update_attr(cache_utils.DynamicCache, 'update', DynamicCache_update), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"
        assert update_attr(cache_utils.DynamicCache, 'get_seq_length', DynamicCache_get_seq_length), f"Unknown DynamicCache definition: {cache_utils.DynamicCache}"


    # In[ ]:


    # ---
    # #### 2.2 Instantiate adapted FP32 model definition
    from tqdm import tqdm
    import torch

    cache_dir = nb_cfg.model.cache_dir
    output_dir = nb_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)


    # In[ ]:


    #======================Configurable setting by users================================
    model_id = nb_cfg.model.model_id
    from transformers import AutoConfig, AutoTokenizer

    llm_config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)

    # Setting context length to be 2048 here, user can change this value to ones' desire (but less than Qwen2' trained contex length)
    context_length = nb_cfg.model.context_length
    # To help with debugging num_hidden_layers could be set to 2 to quickly verify the pipeline and export a two layer model for verification purposes
    if nb_cfg.profiling.qk_layer:
        llm_config.num_hidden_layers = 2

    print(f'num_layer: {llm_config.num_hidden_layers}, context_length : {context_length},'
          f'num_hidden_size :{llm_config.num_attention_heads},  num_kv_heads: {llm_config.num_key_value_heads}')


    # In[ ]:


    #======================Fixed setting that should not be changed by users==============
    # Auto-regression length: number of tokens to consume and number of logits to produce.
    # This value should NOT be changed due to downstream consumption requirements

    setattr(llm_config, 'return_top_k', 0)
    setattr(llm_config, 'return_new_key_value_only', True)
    setattr(llm_config, 'transposed_key_cache', True)
    setattr(llm_config, 'use_combined_mask_input', True)
    setattr(llm_config, 'use_position_embedding_input', True)
    setattr(llm_config, "use_cache", True)
    setattr(llm_config, '_attn_implementation', 'eager')
    setattr(llm_config, '_attn_implementation_internal', 'eager')
    setattr(llm_config, 'use_input_embeddings', True)
    setattr(llm_config, 'mask_neg', nb_cfg.model.mask_neg)
    setattr(llm_config, 'rm_div', True)


    # In[ ]:


    model_name = os.path.basename(model_id).lower()
    model_name = model_name.replace(".", "p").replace("-", "_")


    # In[ ]:


    with event_marker('FP model'):
        model = modeling_qwen2.Qwen2ForCausalLM.from_pretrained(model_id, config=llm_config)
        model.config.return_dict = False
        os.environ['TOKENIZERS_PARALLELISM'] = '0'
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
        ## Adjust the tokenizer to limit to context_length
        tokenizer.model_max_length = context_length


    # In[ ]:


    with event_marker('FP model adaptation for NSP backend completion'):
        for name, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()


    # In[ ]:


    # Loading the calibration data from notebook config
    if nb_cfg.calib.name == 'json':
        device = "cuda:0"
        from llm_utils.qwen2_vl_dataloader import get_qwen2_dataset
        qwen2_dataset_setting = {
            "emb_length": ARN,
            "device": device,
            "qwen2vl_model_id": nb_cfg.model.model_id,
            "calibration_dataset_path": nb_cfg.calib.calibration_dataset_path,
            "ppl_evaluation_dataset_path": nb_cfg.calib.ppl_evaluation_dataset_path,
            "image_dataset_path": nb_cfg.calib.image_dataset_path,
            "vision_input_size": nb_cfg.calib.vision_input_size
        }
        train_dataloader, test_dataloader, dataset = get_qwen2_dataset(model.model, qwen2_dataset_setting, num_test_batches=100)

    elif nb_cfg.calib.name == 'wiki':
        from llm_utils.wikitext_dataloader import get_wiki_dataset
        train_dataloader, test_dataloader, _ = get_wiki_dataset(context_length, tokenizer, cache_dir)
    else:
        raise RuntimeError("Invalid dataset setting from notebook config")


    # In[ ]:


    # ---
    # ### 4. Model Evaluation
    from torch.nn import CrossEntropyLoss
    from llm_utils.forward_pass_wrapper import slice_inputs_and_run_successive_kvcache_inference


    def ppl_eval(model_mode, data_loader, forward_pass_manager, num_batches=0):

        if num_batches == 0:
            num_batches = len(data_loader)
        loss = 0

        for batch_id, batch in enumerate(tqdm(data_loader, total=num_batches, desc="Evaluating")):
            if batch_id >= num_batches:
                break
            if model_mode == "kvcache":
                outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_embeds=batch['input_embeddings'])
            elif model_mode == "bertcache":
                outputs = forward_pass_manager(**batch)
            # outputs = slice_inputs_and_run_successive_kvcache_inference(forward_pass_manager, input_embeds=batch['inputs_embeds'])
            lm_logits = outputs["lm_logits"].cpu()

            # we can either pass input_ids or input_embeds in our fpm, hence with input_embeds we pass the labels.
            if 'input_ids' not in batch:
                batch['input_ids'] = batch['labels']

            lm_logits = lm_logits.reshape(batch['input_ids'].shape[0], -1, lm_logits.shape[-1])
            shift_logits = lm_logits[..., :-1, :].contiguous().to(dtype=torch.float32)
            shift_labels = batch['input_ids'][..., 1:].contiguous().to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss += loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        loss = loss / num_batches
        ppl = loss.exp()

        return ppl


    # In[ ]:


    # change layer 0 &16 attn mask
    def pre_hook_func_0(_, inputs):
        assert len(inputs) == 2
        return (inputs[0], inputs[1] * nb_cfg.model.layer_clamp_0_prefill_mask)
    def pre_hook_func_16(_, inputs):
        assert len(inputs) == 2
        return (inputs[0], inputs[1] * nb_cfg.model.layer_clamp_16_prefill_mask)

    hooks = []
    hooks.append(model.model.layers[0].self_attn.attn_add.register_forward_pre_hook(pre_hook_func_0))
    if not nb_cfg.profiling.qk_layer:
        hooks.append(model.model.layers[16].self_attn.attn_add.register_forward_pre_hook(pre_hook_func_16))


    # In[ ]:


    # ### 4.1 FP32 PPL Eval
    from llm_utils.forward_pass_wrapper import LLMForwardPassManager

    orig_fpm = LLMForwardPassManager(cfg=llm_config, model=model, tokenizer=tokenizer, 
                                     model_mode='kvcache', num_logits_to_return=ARN, separate_tuple_input_output=False,
                                     num_tokens=ARN)

    with event_marker("FP eval"):
        with torch.no_grad():
            with orig_fpm.place_on_device(device):
                orig_ppl = ppl_eval('kvcache', test_dataloader, orig_fpm)

    print(f"ppl score of original fp model: {orig_ppl}")


    # In[ ]:


    from llm_utils.forward_pass_wrapper import get_position_embeddings_from_position_ids, prepare_combined_attention_mask, get_padded_kv_values, flatten_tensors

    def get_dummy_data(model_mode, num_layers, hidden_size, num_attention_heads, rope_theta, tokenizer, device, separate_tuple_input_output, num_kv_heads, num_tokens=None):

        max_tokens = tokenizer.model_max_length
        attention_mask = torch.ones((1, max_tokens), dtype=torch.long, device=device)

        if model_mode == 'bertcache':
            num_tokens = max_tokens

        position_ids = torch.cumsum(attention_mask, dim=1) - 1
        position_ids = position_ids.clip(0, max_tokens - 1)
        position_ids = position_ids[..., :num_tokens]
        position_ids = position_ids.to(device=device)
        past_kv_length = max_tokens - num_tokens if model_mode == 'kvcache' else 0
        attention_mask = prepare_combined_attention_mask(attention_mask, input_shape=(1, num_tokens), past_key_values_length=past_kv_length, 
                                                         device=device, mask_neg=-100)

        position_ids = get_position_embeddings_from_position_ids(position_ids, head_dim=hidden_size // num_attention_heads, 
                                                                 max_length=max_tokens, rope_theta=rope_theta, device=device)
        inputs = {
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            # refer the print values->min_value:-0.10791015625 max_value:0.109375
            'inputs_embeds': torch.empty((1, num_tokens, hidden_size), device=device).uniform_(-0.11, 0.11)
        }

        if model_mode == 'kvcache':
            inputs['past_key_values'] = get_padded_kv_values(past_size=max_tokens - num_tokens,
                                                             num_layers=num_layers,
                                                             hidden_size=hidden_size,
                                                             num_attention_heads=num_attention_heads,
                                                             num_kv_heads=num_kv_heads,
                                                             device=device)

            if separate_tuple_input_output:
                flattened_kvcache = tuple(flatten_tensors(inputs['past_key_values']))
                inputs = inputs['inputs_embeds'], inputs['attention_mask'], inputs['position_ids'][0], inputs['position_ids'][1]
                inputs = inputs + flattened_kvcache
        else:
            if separate_tuple_input_output:
                inputs = inputs['inputs_embeds'], inputs['attention_mask'], inputs['position_ids'][0], inputs['position_ids'][1]
        return inputs


    # In[ ]:


    from aimet_common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel
    import copy

    dummy_input = get_dummy_data('kvcache',
                                 llm_config.num_hidden_layers,
                                 llm_config.hidden_size,
                                 llm_config.num_attention_heads,
                                 llm_config.rope_theta,
                                 tokenizer,
                                 'cpu',
                                 separate_tuple_input_output=False,
                                 num_kv_heads=llm_config.num_key_value_heads,
                                 num_tokens=ARN)


    # In[ ]:


    QNN_SDK_ROOT = '/opt/qcom/aistack/qairt/2.31.0.250130' # insert qnn2.31 path
    assert QNN_SDK_ROOT != '', 'Please point the QNN_SDK_ROOT variable to your QNN SDK'
    sys.path.insert(0, QNN_SDK_ROOT + '/lib/python')
    os.environ['LD_LIBRARY_PATH'] = os.path.join(QNN_SDK_ROOT + '/lib/x86_64-linux-clang', os.getenv('LD_LIBRARY_PATH', ''))


    # In[ ]:


    import aimet_torch.pro.ir_graph_op_handler as ir_graph_op_handler
    from aimet_torch.pro import model_preparer
    from aimet_torch import onnx_utils
    # Setting this flag to False means that the prepared model will be flattened
    # This flag must be set to false because we rely on the model structure being flat to enable weight sharing
    ir_graph_op_handler.KEEP_ORIGINAL_MODEL_STRUCTURE = False

    #configuring the model for KV mode
    model.num_logits_to_return = ARN

    def _get_past_key_values_names(sfx, n_layers):
        all_kvs = []
        for i in range(n_layers):
            all_kvs.append(f'past_key_{i}_{sfx}')
            all_kvs.append(f'past_value_{i}_{sfx}')
        return all_kvs



    # In[ ]:


    input_names = ['inputs_embeds', 'attention_mask', 'position_ids_cos', 'position_ids_sin'] + _get_past_key_values_names('in', llm_config.num_hidden_layers)
    output_names = ['logits'] + _get_past_key_values_names('out', llm_config.num_hidden_layers)

    converter_args_param = ['--input_layout']
    converter_args_value = 'NONTRIVIAL'
    converter_args = []
    for input_param in converter_args_param:
        for input_name in input_names:
            converter_args += [input_param, input_name, converter_args_value]

    # LLM model prepare
    prepare_path = os.path.join(output_dir, 'prepare')
    os.makedirs(prepare_path, exist_ok=True)


    # In[ ]:


    prepare_filename = f'{model_name}_kvcache'

    # if not os.path.exists(os.path.join(prepare_path, f"{prepare_filename}.py")):
    with event_marker("KV MHA Model prepare", flush_ram=True):
        kv_mha_prepared_model = model_preparer.prepare_model(model,
                                                             dummy_input,
                                                             filename=prepare_filename,
                                                             path=prepare_path,
                                                             input_names=input_names,
                                                             output_names=output_names,
                                                             converter_args=converter_args)
    # else:
    #     from aimet_torch.utils import load_pytorch_model
    #     with event_marker("KV MHA Model prepare cached", flush_ram=True):
    #         kv_mha_prepared_model = load_pytorch_model(path=prepare_path, filename=prepare_filename, model_name='ConvertedModel', load_state_dict=True)



    # In[ ]:


    from llm_utils.forward_pass_wrapper import LLMForwardPassManager
    # Calculate ppl score for prepared fp model
    kv_mha_fpm = LLMForwardPassManager(cfg=llm_config, model=kv_mha_prepared_model, tokenizer=tokenizer, model_mode='kvcache', num_logits_to_return=ARN, separate_tuple_input_output=True, num_tokens=ARN)

    with event_marker("Kvcache prepared FP eval"):
        with torch.no_grad():
            with kv_mha_fpm.place_on_device(device):
                prepared_kvcache_ppl = ppl_eval('kvcache', test_dataloader, kv_mha_fpm)
    print(f"ppl score of KV prepared fp model: {prepared_kvcache_ppl}\n"
          f"orig ppl - prepared ppl = {orig_ppl - prepared_kvcache_ppl}")


    # In[ ]:


    import importlib
    import llm_utils.forward_pass_wrapper  # 重新加载模块
    importlib.reload(llm_utils.forward_pass_wrapper)
    from llm_utils.forward_pass_wrapper import LLMForwardPassManager
    sim_fpm = LLMForwardPassManager(
        cfg=llm_config,
        model=copy.deepcopy(kv_mha_prepared_model),  # to avoid creating the sim in_place on the original model
        tokenizer=tokenizer,
        model_mode='kvcache',
        num_logits_to_return=ARN,
        separate_tuple_input_output=True,
        num_tokens=ARN)

    dummy_input = get_dummy_data('kvcache',
                                 llm_config.num_hidden_layers,
                                 llm_config.hidden_size,
                                 llm_config.num_attention_heads,
                                 llm_config.rope_theta,
                                 tokenizer,
                                 device,
                                 separate_tuple_input_output=True,
                                 num_kv_heads=llm_config.num_key_value_heads,
                                 num_tokens=ARN)


    # In[ ]:


    with event_marker("create KVCache Quantsim"):
        with sim_fpm.place_on_device(device):
            quantsim = QuantizationSimModel(model=sim_fpm.model,
                                            quant_scheme=QuantScheme.post_training_tf,
                                            dummy_input=dummy_input,
                                            default_output_bw=16,
                                            default_param_bw=4,
                                            in_place=True,
                                            config_file=htp_config_file)


    # In[ ]:


    from aimet_torch.v2.experimental.quantsim_utils import set_matmul_second_input_producer_to_8bit_symmetric

    set_matmul_second_input_producer_to_8bit_symmetric(quantsim)

    from aimet_torch.v2.experimental import propagate_output_encodings
    import aimet_torch.elementwise_ops as aimet_ops

    propagate_output_encodings(quantsim, aimet_ops.Concat)

    from llm_utils.mixed_precision_overrides import ManualQuantsimMixedPrecisionConfig

    quantsim_adjuster = ManualQuantsimMixedPrecisionConfig(mixed_precision_config_file="config/mixed_precision_config/qwen2_w4a16_gqa.json")
    quantsim_adjuster.apply_exceptions(quantsim)


    # In[ ]:


    from aimet_torch.v2.nn.true_quant import QuantizedConv2d
    from aimet_torch.v2.quantsim.config_utils import set_grouped_blockwise_quantization_for_weights
    import aimet_common.quantsim as qs

    qs.encoding_version = '1.0.0'

    arg = lambda module: isinstance(module, QuantizedConv2d) and module.param_quantizers['weight'].bitwidth == 4
    BLOCK_QUANT_SIZE = 64
    BITWIDTH = 4
    DECOMPRESSED_BITWIDTH = 8

    set_grouped_blockwise_quantization_for_weights(sim = quantsim,
                                                   arg = arg,
                                                   bitwidth = BITWIDTH,
                                                   symmetric = True,
                                                   decompressed_bw = DECOMPRESSED_BITWIDTH,
                                                   block_size = BLOCK_QUANT_SIZE,
                                                   block_grouping = -1)


    # In[ ]:


    def _forward_fn(model, kwargs):
        data_loader = kwargs['data_loader']
        fpm = kwargs['fpm']
        max_iterations = kwargs['num_batches']
        for batch_id, batch in enumerate(tqdm(data_loader, total=max_iterations)):
            if batch_id < max_iterations:
                slice_inputs_and_run_successive_kvcache_inference(fpm, input_embeds=batch['input_embeddings'])
            else:
                break


    # In[ ]:


    from llm_utils.qk_clamp import register_clamp_hooks, update_mask_quantizer
    kwargs = {'data_loader': train_dataloader, 'fpm': sim_fpm, 'num_batches': 100}
    with event_marker("compute encoding", flush_ram=True):
        with sim_fpm.place_on_device("cuda"), register_clamp_hooks(nb_cfg.model.qk_clamp, quantsim):
            quantsim.compute_encodings(_forward_fn, kwargs)


    # In[ ]:


    with event_marker("Sim eval kv mode"):
        with torch.no_grad():
            with sim_fpm.place_on_device(device):
                sim_ppl = ppl_eval("kvcache", test_dataloader, sim_fpm)
    print(f"ppl score of quantsim model: {sim_ppl}\n"
        f"orig ppl - quantsim ppl = {orig_ppl - sim_ppl}")


    # In[ ]:


    # onnx model export
    from aimet_torch.utils import change_tensor_device_placement
    from aimet_torch.onnx_utils import OnnxExportApiArgs
    from aimet_torch import onnx_utils
    from aimet_utils.clip_weights import clip_weights_to_7f7f
    onnx_base_dir = os.path.join(output_dir, 'export')
    onnx_dir = os.path.join(onnx_base_dir, 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)

    onnx_utils.RESTORE_ONNX_MODEL_INITIALIZERS = True

    clip_weights_to_7f7f(quantsim)

    onnx_api_args = OnnxExportApiArgs(opset_version=14, input_names=input_names, output_names=output_names)
    sample_inputs = change_tensor_device_placement(dummy_input, torch.device('cpu'))
    filename_prefix = f"{model_name}_AR{ARN}"
    with event_marker("KVCache export onnx and test vectors", flush_ram=True):
        quantsim.export(onnx_dir, filename_prefix, sample_inputs, onnx_export_args=onnx_api_args, export_model=True)

    from llm_utils.test_vectors import generate_test_vectors

    test_vector_layers = [
        "model_layers_\\d+_input_layernorm_Pow", 
        "model_layers_\\d+_input_layernorm_Cast", 
        "lm_head_conv2d_Conv", "lm_head_MatMul", 
        "model.layers\\d+.input_layernorm.cast", 
        "lm_head_conv", 
        "lm_head"
    ]

    try:
        with event_marker("generate test vector"):
            generate_test_vectors(quantsim, sim_fpm,
                                  train_dataloader,
                                  onnx_base_dir,
                                  num_batches=1,
                                  test_vector_layers=test_vector_layers,
                                  input_names=input_names)
    except Exception as e:
        import traceback
        print(f"[WARNING] generate_test_vectors failed: {e}")
        traceback.print_exc()
        print("[WARNING] Skipping test vectors (not required for Example2)")


    # In[ ]:


    # Export embedding weight
    embeddings = model.model.get_input_embeddings().weight
    print("embedding shape: ", embeddings.shape)
    embeddings.detach().cpu().numpy().tofile(os.path.join(output_dir, 'embedding_weights_151936x1536.raw'))


    # In[ ]:


    # Export tokenizer    
    tokenizer_dir = os.path.join(output_dir, 'tokenizer')
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)


    # In[ ]:







if __name__ == "__main__":
    main()
