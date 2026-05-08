# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------

""" Model adaptions for LoRA. """
from dataclasses import dataclass 
from typing import List, Optional, Union
from onnx.onnx_pb import NodeProto, TensorProto
import copy
from mha2sha.utils.logger import log_info, log_debug
from mha2sha.utils.onnx import get_mul_value
from mha2sha.optimizer_extension.lora_extension import LoraExtension, LoraNode
from mha2sha.utils.encoding_mapper_utils import (
    AimetEncodingVersion,
    create_tensor_name_to_encodings,
    update_encodings_to_v1_format,
    handle_aimet_v1_activ_encodings,
)
LORA_MLP_PATTERN_WILDCARD = ["Transpose", "Reshape"]


class LoraModelAdpation:
    """ Lora model adaption for non-mha2sha adaption. """
    def __init__(self, prequant_adaption) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            prequant_adaption:
                PreQuantAdaption instance holding the model loader and model info. 
        """
        self.prequant_adaption = prequant_adaption
        self.confirmed_lora_alpha_name_list = []
        self.lora_alpha = None
        self.lora_extension = LoraExtension(prequant_adaption)
        self.encodings_map = {
            "activation_encodings": {},
            "param_encodings": {}
        }

    def search_lora_pattern(self, candidate_add_node):
        # quick check (since lora alpha is also an add node)
        if candidate_add_node.name in self.confirmed_lora_alpha_name_list:
            return False
        lora_node = self.lora_extension.verify_and_capture_lora_structure(candidate_add_node)
        if lora_node is None:
            return False
        if lora_node.lora_alpha is not None:
            self.confirmed_lora_alpha_name_list.append(lora_node.lora_alpha.name)

        return lora_node

    def _add_lora_alpha_from_input(self, lora_node):
        """ Re-connect lora tensors to remove redundent nodes """
        lora_node.lora_alpha.input[0] = lora_node.lora_a.output[0]
        lora_node.lora_alpha.input[1] = self.prequant_adaption.lora_alpha_input.name
        lora_node.lora_b.input[0] = lora_node.lora_alpha.output[0]

    def verify_lora_mul_const_input(self, mul_node):
        mul_scale_tensor = mul_node.input[1]

        if mul_scale_tensor in self.prequant_adaption.get_node_by_output_name.keys():
            scale = self.prequant_adaption.get_node_by_output_name[mul_node.input[1]]
        elif mul_scale_tensor in self.prequant_adaption.get_initializer_by_name.keys():
            scale = self.prequant_adaption.get_initializer_by_name[mul_node.input[1]]
        elif mul_scale_tensor in self.prequant_adaption.mha_model_input_names:
            return False
        else:
            raise ValueError(f"expect mul input have type node or initializer.")

        return True

    def add_lora_alpha_from_input(self):
        add_list = [node for node in self.prequant_adaption.model.graph.node if node.op_type == "Add"]
        lora_node_list = []

        for _add in add_list:
            # Search down stream for LORA_MLP_PATTERN
            lora_node = self.search_lora_pattern(_add)
            if lora_node:
                lora_node_list.append(lora_node)
                log_debug(f"find lora: \n{lora_node}")

                if not self.lora_alpha:
                    self.lora_alpha = get_mul_value(
                                            lora_node.lora_alpha,
                                            self.prequant_adaption.get_initializer_by_name,
                                            self.prequant_adaption.get_node_by_output_name
                                      )

        # verify lora mul don't have input from model input.
        lora_node_list = [_lora_node for _lora_node in lora_node_list if self.verify_lora_mul_const_input(_lora_node.lora_alpha)]
        for _lora_node in lora_node_list:
            origin_lora_alpha_name = _lora_node.lora_alpha.input[1]
            self._add_lora_alpha_from_input(_lora_node)
            
            # record encodings map when lora_alpha is a initializer
            if origin_lora_alpha_name in self.prequant_adaption.get_initializer_by_name:
                activation_enc_map = self.encodings_map["activation_encodings"]
                new_lora_alpha_name = _lora_node.lora_alpha.input[1]
                if origin_lora_alpha_name not in self.encodings_map:
                    activation_enc_map[origin_lora_alpha_name] = [new_lora_alpha_name]
                elif new_lora_alpha_name not in self.encodings_map[origin_lora_alpha_name]:
                    activation_enc_map[origin_lora_alpha_name].append(new_lora_alpha_name)
        
        log_info(f"Updated {len(lora_node_list)} lora to use alpha from model input.")
    
    def transform_encodings(self, origin_encodings):
        encoding_version = AimetEncodingVersion(origin_encodings["version"])

        origin_name_to_encodings = origin_encodings
        if encoding_version == AimetEncodingVersion.V1:
            origin_name_to_encodings = create_tensor_name_to_encodings(
                                            origin_name_to_encodings)
        
        new_name_to_encodings = copy.deepcopy(origin_name_to_encodings)
        
        # add new encodings / modify encodings
        for enc_type in ("activation_encodings", "param_encodings"):
            for origin_name, new_names in self.encodings_map[enc_type].items():
                if origin_name in origin_name_to_encodings[enc_type]:
                    curr_enc = origin_name_to_encodings[enc_type][origin_name]
                else:
                    # mha encodings may not have activation encodings for lora-alpha.
                    # this is a normal case, so ignore it.
                    log_info(
                        f"cannot find encodings of '{origin_name}', " + 
                        "so ignore it in prequant encodings transformation"
                    )
                    continue
                
                if len(new_names) != 1:
                    raise RuntimeError("currently only one-to-one mapping is supported " +
                                       "for prequant-adaption")
                
                if encoding_version == AimetEncodingVersion.V1:
                    # even for param encodings, we don't need to split them in this stage, 
                    # so calling handle_aimet_v1_activ_encodings is fine
                    curr_enc = handle_aimet_v1_activ_encodings(curr_enc, new_names[0])

                new_name_to_encodings[enc_type][new_names[0]] = curr_enc

        if encoding_version == AimetEncodingVersion.V1:
            _key_values_not_mapped = {
                key: origin_encodings[key]
                for key in origin_encodings.keys()
                if key not in ["param_encodings", "activation_encodings"]
            }
            new_name_to_encodings = update_encodings_to_v1_format(new_name_to_encodings, 
                                                                  _key_values_not_mapped)

        # note: unused encodings are not removed, not mha2sha stage will remove unused encodings
        return new_name_to_encodings
