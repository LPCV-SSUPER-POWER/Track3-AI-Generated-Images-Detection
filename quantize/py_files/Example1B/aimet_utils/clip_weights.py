# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import torch
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.quantization.affine.quantizer import AffineQuantizerBase


def clip_weights_to_7f7f(sim: 'QuantizationSimModel'):
    """
    Clip sim model weights which are 16 bit symmetric to have a max of 0x7f7f when quantized.
    :param sim: Quantsim model to clip weights for
    """
    affected_layers = []
    for name, quant_layer in sim.quant_wrappers():
        # pylint: disable=too-many-boolean-expressions
        if 'weight' in quant_layer.param_quantizers and \
                quant_layer.param_quantizers['weight'] is not None and \
                quant_layer.param_quantizers['weight'].bitwidth == 16 and \
                isinstance(quant_layer.param_quantizers['weight'], AffineQuantizerBase) and \
                quant_layer.param_quantizers['weight'].symmetric and \
                quant_layer.param_quantizers['weight'].is_initialized():
            clipped_weight = torch.minimum(quant_layer.weight,
                                           quant_layer.param_quantizers['weight'].get_scale() * 0x7f7f)
            with torch.no_grad():
                quant_layer.weight.copy_(clipped_weight)

            affected_layers.append(name)
    logger_str = f'Clipping weights of the following layers to 0x7f7f max quantized value: {affected_layers}'
    print(logger_str)