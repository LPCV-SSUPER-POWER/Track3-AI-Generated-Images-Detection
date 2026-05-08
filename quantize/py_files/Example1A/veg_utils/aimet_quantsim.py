# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright 2024 Qualcomm Technologies, Inc. All rights reserved.
#  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")
#
#  The party receiving this software directly from QTI (the "Recipient")
#  may use this software as reasonably necessary solely for the purposes
#  set forth in the agreement between the Recipient and QTI (the
#  "Agreement"). The software may be used in source code form solely by
#  the Recipient's employees (if any) authorized by the Agreement. Unless
#  expressly authorized in the Agreement, the Recipient may not sublicense,
#  assign, transfer or otherwise provide the source code to any third
#  party. Qualcomm Technologies, Inc. retains all ownership rights in and
#  to the software
#
#  This notice supersedes any other QTI notices contained within the software
#  except copyright notices indicating different years of publication for
#  different portions of the software. This notice does not supersede the
#  application of any third party copyright notice to that third party's
#  code.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

import torch
import os
from aimet_torch import elementwise_ops
from aimet_torch.qc_quantize_op import QcQuantizeWrapper
from aimet_torch.onnx_utils import OnnxExportApiArgs
from aimet_torch.utils import change_tensor_device_placement
from aimet_torch.pro.model_preparer import prepare_model as prepare_model_pro
from aimet_torch.pro.quantsim import QuantizationSimModel
from aimet_torch.quantsim import load_encodings_to_sim

from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_common.defs import QuantizationDataType

from PIL import Image
import torch
import copy
import os

from glob import glob
from tqdm import tqdm
import random
import json


def layernorm_exceptions(quant_sim):
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, torch.nn.LayerNorm):
            print("Setting", name, "param_quantizers to int16")
            module.param_quantizers['weight'].enabled = True
            module.param_quantizers['weight'].bitwidth = 16


def matmul_exceptions(quant_sim):
    for name, module in quant_sim.model.named_modules():
        if isinstance(module, QcQuantizeWrapper) and isinstance(module._module_to_wrap, elementwise_ops.MatMul):
            print("Setting", name, "second input quantizer to 8bit symmetric")
            module.input_quantizers[1].enabled = True
            module.input_quantizers[1].bitwidth = 8
            module.input_quantizers[1].use_symmetric_encodings = True


def enable_concat_input_quantizers(quant_sim):
    # quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.input_quantizers[0].enabled = True
    # quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.input_quantizers[0].bitwidth = 16
    # quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.input_quantizers[1].enabled = True
    # quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.input_quantizers[1].bitwidth = 16
    quant_sim.model.patch_embed_proj_Unsqueeze.input_quantizers[0].enabled = True
    quant_sim.model.patch_embed_proj_Unsqueeze.input_quantizers[0].bitwidth = 16
    quant_sim.model.patch_embed_proj_Unsqueeze.input_quantizers[1].enabled = True
    quant_sim.model.patch_embed_proj_Unsqueeze.input_quantizers[1].bitwidth = 16


def buffer_concat_exceptions(quant_sim):
    print("Matching input[0] and [1] encodings for", "patch_embed_proj_Unsqueeze")
    # quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.input_quantizers[0] = quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.output_quantizers[0]
    #quant_sim.model.vision_tower_vision_model_embeddings_Expand_output_0_nontrivial.input_quantizers[0] = quant_sim.model.vision_tower_vision_model_embeddings_concat_Concat.output_quantizers[0]
    quant_sim.model.patch_embed_proj_Unsqueeze.input_quantizers[0] = quant_sim.model.patch_embed_proj_Unsqueeze.output_quantizers[0]


def get_sqnr(org_out, quant_out, in_db=True, eps=0.0):
    quant_error = org_out - quant_out
    exp_noise = quant_error.pow(2).view(quant_error.shape[0], -1).mean(1) + eps
    exp_signal = org_out.pow(2).view(org_out.shape[0], -1).mean(1)
    sqnr = (exp_signal / exp_noise).mean()
    sqnr_db = 10 * torch.log10(sqnr)
    return sqnr_db if in_db else sqnr


def compute_sqnr(preprocessed_images, veg_fp, veg_qt, pos_emb, attention_mask):
    fp_outs, qt_outs = [], []
    with torch.no_grad():
        for i, img in tqdm(enumerate(preprocessed_images)):
            embeddings_fp = veg_fp(img, pos_emb[0], pos_emb[1], attention_mask)
            fp_outs.append(embeddings_fp)
            embeddings_qt = veg_qt(img, pos_emb[0], pos_emb[1], attention_mask)
            qt_outs.append(embeddings_qt)
    sqnr = get_sqnr(torch.stack(fp_outs).cpu(), torch.stack(qt_outs).cpu())
    return sqnr
