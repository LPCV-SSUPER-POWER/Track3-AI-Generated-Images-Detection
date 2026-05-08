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
from onnx.onnx_pb import ModelProto, NodeProto, TensorProto
from onnx import helper
import json
import os, copy


from mha2sha.utils.clean import clean_model, topological_sort

from mha2sha.utils.onnx import (
    get_initializer_mappings,
    get_node_by_input_name,
    get_node_by_output_name,
    get_node_mappings,
    get_value_info_proto_mapping,
)

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning
from mha2sha.utils.utils import update_all_mapping_dicts

PREQUANT_ENCODING_MAP_PATH="prequant_encodings_map.json"



class PreQuantAdaption:
    """
    PreQuantAdaption class is responsible for making pre-quant model adaptions up to mha2sha. 
    """
    def __init__(
        self,
        model: ModelProto,
        lora_model: bool,
        lora_alpha_from_input: bool,
    ):
        """
        Initialization

        :param model: ModelProto
        :param lora_model: Is lora model
        :param lora_alpha_from_input: Is lora alpha from model input.
        :param mha_conv: is mha use conv instead of matmul in projection
        """

        self.model = model
        self.lora_model = lora_model
        self.lora_alpha_from_input = lora_alpha_from_input
        self.encodings_map = {
            "activation_encodings": {},
            "param_encodings": {}
        }
        self._update_all_mapping_dicts()  # Initial mapping

    def _update_all_mapping_dicts(self):
        """Helper function to update mappings to nodes.

        Updates all the mapping dictionaries such as `get_initializer_by_name`. These need
        to be updated as nodes are added to the graph and are not yet know.
        """
        update_all_mapping_dicts(self)

    def optimize(self, mha_encodings, enc_map_save_dir):
        if self.lora_model and self.lora_alpha_from_input:
            from mha2sha.transformations.lora_model_adaption import LoraModelAdpation
            log_info("Running model adaption for --lora-alpha-from-input outside mha.")       
            # Add lora alpha to model input
            self.lora_alpha_input = helper.make_tensor_value_info(
                                        "lora_alpha",
                                        TensorProto.FLOAT,
                                        [1]
                                        )
            self.model.graph.input.append(self.lora_alpha_input)

            # Apply lora model adaptions
            self._update_all_mapping_dicts()
            self.lora_model_adaption = LoraModelAdpation(self)
            self.lora_model_adaption.add_lora_alpha_from_input()
            self.lora_alpha = self.lora_model_adaption.lora_alpha
            self.encodings_map = self.lora_model_adaption.encodings_map
            
            if mha_encodings is not None:
                mha_encodings = self.lora_model_adaption.transform_encodings(
                                                                mha_encodings
                                                         )
        self.model = topological_sort(clean_model(self.model))
        self._update_all_mapping_dicts()

        self.save_encodings_map(enc_map_save_dir)
        
        return self.model, mha_encodings

    def save_encodings_map(self, dir_path):
        with open(os.path.join(dir_path, PREQUANT_ENCODING_MAP_PATH), "w") as f:
            json.dump(self.encodings_map, f, indent=4)
    