import torch
import contextlib
import inspect
import math
import re
from aimet_torch.v2.nn import BaseQuantizationMixin, custom
from aimet_torch.v2.nn.true_quant import QuantizationMixin
from aimet_torch.quantsim_config.builder import LazyQuantizer



def create_new_hook(a: int, b: int):
    def hook(c, d):
        return torch.clamp(*d, a, b)
    return hook

@contextlib.contextmanager
def register_hooks(list1: list, list2: list, list3: list):
    k = []
    try:
        for l, m in enumerate(list1):
            handle = m.register_forward_pre_hook(create_new_hook(list2[l], list3[l]))
            k.append(handle)
        yield
    finally:
        for h in k:
            h.remove()

@contextlib.contextmanager
def register_clamp_hooks(qk_clamp_cfg, quantsim):
    qk_list, layer_idxs, hook_cfg = [],[],{}
    for c in qk_clamp_cfg:
        layer_idxs.append(c['layer_idx'])
        hook_cfg[c['layer_idx']] = c

    for name, module in quantsim.model.named_modules():
        # model_layers_0_self_attn_q_proj_conv_Conv.output_quantizers.0
        # model_layers_0_self_attn_k_proj_conv_Conv.output_quantizers.0
        # model_layers_16_self_attn_q_proj_conv_Conv.output_quantizers.0
        # model_layers_16_self_attn_k_proj_conv_Conv.output_quantizers.0
        rs = re.findall(r'\d+', name)
        if len(rs)<=0: continue
        layer_idx = int(rs[0])
        if not layer_idx in layer_idxs: continue
        if not (re.search('output', name) and name[-2:] == '.0'): continue

        is_k = re.search('k_proj', name) # k
        is_q =  re.search('q_proj', name) # q

        if not (is_k or is_q): continue

        min_max = hook_cfg[layer_idx]['k_min_max_val'] if is_k else hook_cfg[layer_idx]['q_min_max_val']
        qk_list.append({"module":module, "min":min_max[0], "max":min_max[1] })

    handles = []
    try:
        for i, item in enumerate(qk_list):
            handle = item["module"].register_forward_pre_hook(create_new_hook(item["min"], item["max"]))
            handles.append(handle)
        yield
    finally:
        for h in handles:
            h.remove()


class Null(custom.Reshape):
    # pylint:disable=arguments-differ
    def forward(self, input1, input2):
        if torch.jit.is_tracing():
            return super().forward(input1, input2)
        return super().forward(input1, input1.shape)


class GoodAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.null = QuantizationMixin.from_module(Null())
        self.add = QuantizationMixin.from_module(custom.Add())

    def forward(self, input1, input2):
        bsz, _, len1, len2 = input1.shape
        return self.add(input1, self.null(input2, [bsz, -1, len1, len2]))


@QuantizationMixin.implements(Null)
class QuantizedNull(QuantizationMixin, Null):
    # pylint: disable=arguments-differ
    def forward(self, x, y):
        if self.input_quantizers[0]:
            x = self.input_quantizers[0](x)

        with self._patch_quantized_parameters():
            out = super().forward(x, y)

        if self.output_quantizers[0]:
            out = self.output_quantizers[0](out)

        return out


def update_module(q, t):
    q.encoding_analyzer.reset_stats()
    batch_statistics = q.encoding_analyzer.update_stats(t)
    num_steps = math.pow(2, q.bitwidth) - 1
    dynamic_min, dynamic_max =\
                q.encoding_analyzer.compute_encodings_from_stats(batch_statistics,
                                                                 num_steps,
                                                                 q.symmetric)
    dynamic_min = dynamic_min.to(dtype=q.min.dtype,
                                    device=q.min.device).expand_as(q.min)
    dynamic_max = dynamic_max.to(dtype=q.max.dtype,
                                    device=q.max.device).expand_as(q.max)
    q.set_range(dynamic_min, dynamic_max)


def update_mask_quantizer(quantsim, input_names,llm_config):
    model_name = quantsim.connected_graph._model_name # pylint: disable=protected-access
    quant_modules = {name: module for name, module in quantsim.model.named_modules()
                        if isinstance(module, BaseQuantizationMixin)}

    names = list(inspect.signature(quantsim.model.forward).parameters)
    names2, values1, values2 = [], [], []
    for name, module in quant_modules.items():
        if isinstance(module, custom.Add):
            find_key = quantsim.connected_graph._module_to_op_dict[quantsim.connected_graph._name_to_module[f'{model_name}.{name}']]
            if find_key.inputs[1].is_model_input and \
                names[int(re.findall(r'\d+', find_key.inputs[1].name)[0])] == input_names[1]:
                r_mo = GoodAdd()
                r_mo.null.input_quantizers[0] = LazyQuantizer(module.input_quantizers[1].bitwidth,
                                                            quantsim._rounding_mode,
                                                            quantsim._quant_scheme,
                                                            module.input_quantizers[1].symmetric,
                                                            enabled_by_default=True
                                                            ).realize()
                r_mo.null.output_quantizers[0] = LazyQuantizer(module.input_quantizers[1].bitwidth,
                                                            quantsim._rounding_mode,
                                                            quantsim._quant_scheme,
                                                            module.input_quantizers[1].symmetric,
                                                            enabled_by_default=True
                                                            ).realize()
                r_mo.add.output_quantizers[0] = module.output_quantizers[0]
                setattr(quantsim.model, name, r_mo)
                if module.input_quantizers[1].is_initialized() and module.output_quantizers[0].is_initialized():
                    names2.append(name)
                    value1 = -abs(module.output_quantizers[0].max - module.output_quantizers[0].min)
                    values1.append(llm_config.mask_min if value1 < llm_config.mask_min else value1)
                    values2.append(module.input_quantizers[1].max)
                else:
                    raise AttributeError


    if names2:
        value3 = llm_config.mask_min
        for name, value1, value2 in zip(names2, values1, values2):
            update_module(quantsim.model.get_submodule(name).null.input_quantizers[0], torch.tensor([value3, value2]))
            update_module(quantsim.model.get_submodule(name).null.output_quantizers[0], torch.tensor([value1, value2]))
