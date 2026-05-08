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
from collections import defaultdict
from dataclasses import dataclass 
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, Union
from onnx import numpy_helper, helper
from onnx.onnx_pb import NodeProto, TensorProto

from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning

from mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_up_based_on_cond,
    get_next_node_down_based_on_cond,
    get_least_commom_ancestor_with_verified_pathway,
    get_mul_value,
)
from mha2sha.utils.utils import BranchType, ExtensionTypes
from mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict
)

mha2sha_hf_model_optimizer = Any  # Causes circular import

LORA_BRANCH_PREFIX = [ExtensionTypes.LORA+"_"+qkv for qkv in ["q", "k", "v"]]
LORA_ACTIVATION_ENCODING_KEYS = [
    "lora_b",
    "lora_alpha",
    "lora_add",
]
LORA_PARAM_ENCODING_KEYS=[
    "lora_b_param"
]

@dataclass
class LoraNode:
    """ Defines nodes to record for lora models. """
    lora_a: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_b: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_add: Optional[Union[NodeProto, List[NodeProto]]] = None
    lora_alpha: Optional[Union[NodeProto, List[NodeProto]]] = None
    base_linear: Optional[Union[NodeProto, List[NodeProto]]] = None

    def __str__(self):
        # useful for printing and debuging
        return f"LoraNode(\n" \
               f"    lora_a: '{self.lora_a.name if self.lora_a else None}',\n" \
               f"    lora_b: '{self.lora_b.name if self.lora_b else None}',\n" \
               f"    lora_add: '{self.lora_add.name if self.lora_add else None}',\n" \
               f"    lora_alpha: '{self.lora_alpha.name if self.lora_alpha else None}',\n" \
               f"    base_linear: '{self.base_linear.name if self.base_linear else None}'\n" \
               f")"
    
    def __repr__(self):
        return str(self)

def create_branch_lora_encoding_mapping_dict(mha_lora_dict, prefix):
    """ create Dict[str, NodeMappingDict] for lora nodes. """
    return {
        f"{prefix}_lora_b": create_activation_node_mapping_dict(
                input_1_name = mha_lora_dict.lora_b.input[0],
                output_name = mha_lora_dict.lora_b.output[0]
            ),
        f"{prefix}_lora_add": create_activation_node_mapping_dict(
                mha_lora_dict.lora_add.input[0],
                mha_lora_dict.lora_add.input[1],
                mha_lora_dict.lora_add.output[0]
            ),
        f"{prefix}_lora_alpha": create_activation_node_mapping_dict(
                mha_lora_dict.lora_alpha.input[0],
                mha_lora_dict.lora_alpha.input[1],
                mha_lora_dict.lora_alpha.output[0]
            ),
        f"{prefix}_lora_b_param": NodeMappingDict(
                mha_mapping_name_list = ["mha_param_name"],
                sha_mapping_name_list = ["sha_param_name"],
                mapping_name_dict = {
                    "mha_param_name": mha_lora_dict.lora_b.input[1],
                    "sha_param_name": None
                    }
            )
        }

def create_lora_encoding_mapping_dict(mha_lora_dict_list: List[LoraNode]):
    """
    Create a Dict[str, NodeMappingDict] for lora nodes which will be used in EncodingMapper.
    :param mha_rope_dict: Collected MHA rope input outpue tensor names in optimizer.
    :return LoraEncodingMappingDict:
    """
    # mha_rope_dict can be None when MHA ROPE pattern match fail.
    lora_dict = {}
    for mha_lora_dict, prefix in zip(mha_lora_dict_list, LORA_BRANCH_PREFIX): 
        if mha_lora_dict is not None:
            _lora_dict = create_branch_lora_encoding_mapping_dict(mha_lora_dict, prefix)
            lora_dict.update(_lora_dict)

    return lora_dict


def update_lora_sha_encoding_name_to_lora_encoding_mapping_dict(
        lora_encoding_mapping_dict,
        q_sha_lora_node,
        k_sha_lora_node,
        v_sha_lora_node
    ):
    """
    Update sha LORA encoding names to EncodingMappingDict.
    :param lora_encoding_mapping_dict: encoding_mapping_dict.lora: Dict[str, NodeMappingDict]
    :param q_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param k_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    :param v_sha_lora_node: A name dict for sha nodes created when optimizer split and create nodes for sha
    """
    for lora_branch_prefix, lora_sha_node in zip(LORA_BRANCH_PREFIX, [q_sha_lora_node, k_sha_lora_node, v_sha_lora_node]):

        # Check lora b for exsitance. Empty lora_b list should retuen False.
        if lora_sha_node.lora_b:
            # Handle activation encodings
            for lora_node_name in LORA_ACTIVATION_ENCODING_KEYS:
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict = lora_encoding_mapping_dict[lora_branch_prefix+"_"+lora_node_name],
                    sha_node_list = getattr(lora_sha_node, lora_node_name)
                )
            
            for lora_node_name in LORA_PARAM_ENCODING_KEYS:
                update_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict = lora_encoding_mapping_dict[lora_branch_prefix+"_"+lora_node_name],
                    sha_node_list = getattr(lora_sha_node, "lora_b")
                )

class LoraExtension:
    """ Extenstion helpers for mha2sha_optimzer to bridge Morpheus pipeline code base and v1.0.0 release.  """
    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info. 
        """
        self.mha2sha_optim = mha2sha_optim
        self.map_lora_encoding = True

    def reset_sha_encoding_name_list(self):
        """
        Reset mha sha names for LORA tensors.
        """
        self.map_lora_encoding = True
        self.q_sha_lora_node = LoraNode(
                                    lora_b=[],
                                    lora_add=[],
                                    lora_alpha=[]
                                )
        self.k_sha_lora_node = LoraNode(
                                    lora_b=[],
                                    lora_add=[],
                                    lora_alpha=[]
                                )
        self.v_sha_lora_node = LoraNode(
                                    lora_b=[],
                                    lora_add=[],
                                    lora_alpha=[]
                                )

    def update_sha_lora_node(self, 
                             branch_type,
                             lora_b,
                             lora_alpha,
                             lora_add):
        """ Update sha node """
        if branch_type == BranchType.Q:
            sha_lora_node = self.q_sha_lora_node
        elif branch_type == BranchType.K:
            sha_lora_node = self.k_sha_lora_node
        elif branch_type == BranchType.V:
            sha_lora_node = self.v_sha_lora_node

        sha_lora_node.lora_b.append(lora_b)
        sha_lora_node.lora_add.append(lora_add)
        sha_lora_node.lora_alpha.append(lora_alpha)

    def verify_and_capture_lora_structure(self, add_lora):
        """
        Find base linear, lora_b and lora_b.
            x-----------
            |           |
            |         lora_a
            |           |
        base linear   Mul alpha
            |           |
            |         lora_b
            |           |
            Add--------/
        
        the lora-structure will be verified.
        If the verification failed, None will be returned.

        combine "lora verification" and "lora nodes matching" into one function
        since the majority of their logic is identical.
        """

        matmul_op_type = "Conv"

        # add_lora's inputs should come from two different nodes
        prev_node_0 = self.mha2sha_optim.get_node_by_output_name.get(
                                add_lora.input[0], None)
        prev_node_1 = self.mha2sha_optim.get_node_by_output_name.get(
                                add_lora.input[1], None)
        if prev_node_0 is None or prev_node_1 is None:
            return None
        if prev_node_0.name == prev_node_1.name:
            return None

        # we only allow transpose / reshape
        # between "Add" and matmul_op_type
        allow_types = set(["Transpose", "Reshape"])
        allow_types.add(matmul_op_type)

        try:
            temp_base_linear = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[add_lora.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type,
                node_end_search_cond=lambda n: n.op_type not in allow_types
            )
        except NodeNotFoundError:
            return None 

        try:
            temp_lora_b = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[add_lora.input[1]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type,
                node_end_search_cond=lambda n: n.op_type not in allow_types
            )
        except NodeNotFoundError:
            return None

        temp_base_linear_init = numpy_helper.to_array(
                self.mha2sha_optim.get_initializer_by_name[temp_base_linear.input[1]])
        temp_lora_b_init = numpy_helper.to_array(
                self.mha2sha_optim.get_initializer_by_name[temp_lora_b.input[1]])

        # base linear input should have higher rank then lora_b input
        # ONNX conv weight is [O, I, kH, kW]
        if temp_base_linear_init.shape[1] > temp_lora_b_init.shape[1]:
            base_linear = temp_base_linear
            lora_b = temp_lora_b
        else:
            base_linear = temp_lora_b
            lora_b = temp_base_linear

        # we only allow transpose / reshape / Mul
        # between matmul_op_type and matmul_op_type
        allow_types = set(["Mul", "Transpose", "Reshape"])
        allow_types.add(matmul_op_type)

        if lora_b.input[0] not in self.mha2sha_optim.get_node_by_output_name.keys():
            return None

        try:
            lora_a = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[lora_b.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == matmul_op_type,
                node_end_search_cond=lambda n: n.op_type not in allow_types
            )
        except NodeNotFoundError:
            return None

        # verify lora_a and base_linear has a LCA (least common ancestor)
        # we only allow transpose / reshape
        # - between LCA and lora_a
        # - between LCA and base_linear
        # that is we only allow transpose / reshape as pathway nodes
        allow_types = set(["Transpose", "Reshape"])
        lca = get_least_commom_ancestor_with_verified_pathway(
                    self.mha2sha_optim.get_node_by_output_name[lora_a.input[0]],
                    self.mha2sha_optim.get_node_by_output_name[base_linear.input[0]],
                    self.mha2sha_optim,
                    pathway_nodes_verifier=lambda n:n.op_type in allow_types
            )
        if lca is None:
            log_debug(
                "Ignoring candiate lora structure, reason: LCA not found\n    "
                f"base_linear:'{base_linear.name}', "
                f"lora_a:'{lora_a.name}', "
                f"lora_b:'{lora_b.name}'")
            return None

        lora_node = LoraNode(
            base_linear=base_linear,
            lora_a=lora_a,
            lora_b=lora_b,
            lora_add=add_lora
        )

        # caputre lora alpha
        lora_alpha = get_next_node_down_based_on_cond(
            lora_a,
            self.mha2sha_optim.get_node_by_input_name,
            node_found_cond=lambda n: n.op_type == "Mul",
            node_end_search_cond=lambda n: n == lora_b
        )

        if lora_alpha:
            lora_node.lora_alpha = lora_alpha
            # verify there is only one Mul between lora_a and lora_b
            lora_alpha_up = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[lora_b.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Mul"
            )

            if lora_alpha is not lora_alpha_up:
                log_warning(
                    "Ignoring candiate lora structure, "
                    "reason: more than one lora-Mul\n    "
                    f"base_linear:'{base_linear.name}', "
                    f"lora_a:'{lora_a.name}', "
                    f"lora_b:'{lora_b.name}'")
                return None

            # verify one input of this Mul is constant or input
            def is_tensor_cst_or_input(tensor_name):
                if tensor_name in self.mha2sha_optim.mha_model_input_names:
                    return True # input
                producer = self.mha2sha_optim.get_node_by_output_name.get(tensor_name, None)
                if producer and (producer.op_type in ("Constant", "Identity")):
                    return True # Constant or Identity
                if tensor_name in self.mha2sha_optim.get_initializer_by_name.keys():
                    return True # initializer (same as consant)
                return False
            if np.array(
                    [is_tensor_cst_or_input(x) for x in lora_alpha.input]
                ).sum() != 1:
                log_warning(
                    "Ignoring candiate lora structure, "
                    "reason: one of lora-Mul is neither constant nor input\n    "
                    f"base_linear:'{base_linear.name}', "
                    f"lora_a:'{lora_a.name}', "
                    f"lora_b:'{lora_b.name}'")
                return None

        return lora_node

    def get_qkv_lora_structure(self, qk_matmul_node, qkv_matmul_node):
        """
        Find Add op adds lora base linear and lora adaptor. Search up from qk_matmul or qkv_matmul 
        for any Conv/MatMul, then search down for and Add between Conv and qk/qkv_matmul. Varify 
        founded lora adds.
        :return q_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        :return k_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        :return v_add_lora: lora_add if verifed, proj Conv/MatMul if not a lora branch
        """
        def is_elementwise_Add(node):
            # element-wise add
            return node.op_type == "Add" and node.input[1] not in self.mha2sha_optim.get_initializer_by_name.keys()

        proj_op_type = "Conv" if self.mha2sha_optim.mha_conv else "MatMul"

        # - when lora exists, q_conv_candidate may be lora's conv or base_linear's conv
        #   but that dosen't matter
        # - when lora dosen't exist, q_conv_candidate is the base_linear's conv
        qkv_convs = dict()
        # defaultdict in the case we never set the lora nodes, we get None at the end
        qkv_lora_nodes = defaultdict(lambda: None)
        qkv_branch_types = [BranchType.Q, BranchType.K, BranchType.V]

        for branch_type in qkv_branch_types:
            found_lora_node = False
            qk_matmul_input_index = 0 if branch_type == BranchType.Q else 1
            qk_or_qkv_matmul_node = (
                qk_matmul_node
                if branch_type in [BranchType.Q, BranchType.K]
                else qkv_matmul_node
            )

            try:
                conv_candidate = get_next_node_up_based_on_cond(
                    self.mha2sha_optim.get_node_by_output_name[
                        qk_or_qkv_matmul_node.input[qk_matmul_input_index]
                    ],
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == proj_op_type,
                )

                add_lora = get_next_node_down_based_on_cond(
                    conv_candidate,
                    self.mha2sha_optim.get_node_by_input_name,
                    node_found_cond=is_elementwise_Add,
                    node_end_search_cond=lambda n: n == qk_or_qkv_matmul_node,
                )

                lora_node = self.verify_and_capture_lora_structure(add_lora)
                qkv_lora_nodes[branch_type] = (
                    lora_node  # will be None or the LoRA node
                )

                if lora_node is not None:
                    found_lora_node = True
            except NodeNotFoundError:
                # found_lora_node starts as False so we don't need to do anything for conv
                # and defaultdict handles lora_node to default to None when we return
                ...

            if not found_lora_node:
                log_warning(f"No lora adaptor found on branch {branch_type.name}")
                qkv_convs[branch_type] = conv_candidate
            else:
                qkv_convs[branch_type] = lora_node.base_linear

        return [qkv_convs[b] for b in qkv_branch_types] + [qkv_lora_nodes[b] for b in qkv_branch_types]


    def update_dqkv_for_lora(self, dqkv, lora_node):
        """
        Get dquery, dkey, dvalue in info dict for lora.
        """
        lora_b = lora_node.lora_b
        lora_b_init = self.mha2sha_optim._mha_conv_extension.get_conv_weight_in_OI(lora_b)

        dqkv["lora_b_init"] = lora_b_init
        dqkv["mha_lora_node"] = lora_node
        return dqkv

    def get_qkv_info(
            self,
            qk_matmul_node: NodeProto,
            qkv_matmul_node: NodeProto,
        ) -> Tuple[Dict, Dict, Dict]:
        """
            Function responsible for collecting QKV information.

            :param qk_matmul_node: The MatMul of where the Query and Key branches join.
            :param qkv_matmaul_node: The MatMul of where the Query, Key, and Key branches join.

            :return dquery: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            :return dkey: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            :return dvalue: Dict - it contains matmul (node, initializers)
                    and add (node, initializers).
            """            
        # Find Add op that adds up baselinear and lora output
        assert self.mha2sha_optim.mha_conv, "Support mha-conv lora at the moment"

        q_conv, k_conv, v_conv, \
            q_lora_node, k_lora_node, v_lora_node \
                = self.get_qkv_lora_structure(qk_matmul_node, qkv_matmul_node)

        dquery = self.mha2sha_optim._mha_conv_extension.get_dqkv_for_qkv_info(q_conv)
        if q_lora_node is not None:
            dquery = self.update_dqkv_for_lora(dquery, q_lora_node)

        dkey = self.mha2sha_optim._mha_conv_extension.get_dqkv_for_qkv_info(k_conv)
        if k_lora_node is not None:
            dkey = self.update_dqkv_for_lora(dkey, k_lora_node)

        dvalue = self.mha2sha_optim._mha_conv_extension.get_dqkv_for_qkv_info(v_conv)
        if v_lora_node is not None:
            dvalue = self.update_dqkv_for_lora(dvalue, v_lora_node)

        return dquery, dkey, dvalue

    def get_single_qkv_lora_input(self, dqkv):
        """ if lora_a in dqkv, add lora alpha and return alpha mul, else return None. """
        if "mha_lora_node" in dqkv.keys():
            mha_lora_node = dqkv["mha_lora_node"]
            lora_inp = mha_lora_node.lora_a

            # Reuse lora input [1] if it is model input
            if mha_lora_node.lora_alpha.input[1] in self.mha2sha_optim.mha_model_input_names:
                lora_inp = self.mha2sha_optim._op_factory.get_element_mul_op(
                            lora_inp,
                            mha_lora_node.lora_alpha.input[1]
                        )
            else:
                lora_inp, lora_alpha_init = self.mha2sha_optim._op_factory.get_mul_op(
                                                lora_inp,
                                                get_mul_value(mha_lora_node.lora_alpha,
                                                    self.mha2sha_optim.get_initializer_by_name,
                                                    self.mha2sha_optim.get_node_by_output_name
                                                )
                                            )
                self.mha2sha_optim.model.graph.initializer.extend(lora_alpha_init)

            self.mha2sha_optim.model.graph.node.append(lora_inp)
            return lora_inp

        return None

    def get_qkv_lora_input(self, info_dict):
        """ Return query_lora_inp """
        query_lora_inp = self.get_single_qkv_lora_input(info_dict["query"])
        key_lora_inp = self.get_single_qkv_lora_input(info_dict["key"])
        value_lora_inp = self.get_single_qkv_lora_input(info_dict["value"])

        return query_lora_inp, key_lora_inp, value_lora_inp

    def attach_single_lora_adaptor(self,
                                   branch_info_dict,
                                   ns,
                                   head_num,
                                   head_dim,
                                   inp,
                                   lora_inp,
                                   branch_type,
                                   ):

        """ Attach lora adpator from lora out to one of qkv conv """
        lora_out = None
        init_name = "lora_b_init"
        if init_name in branch_info_dict and \
                (conv_weight_init := branch_info_dict[init_name]) is not None:
            lora_out = self.mha2sha_optim._mha_conv_extension.\
                        create_single_conv(
                            conv_weight_init,
                            ns,
                            head_num,
                            head_dim,
                            lora_inp,
                            suffix="lora_b",
                            branch_type=branch_type,
                            # lora has no bias
                        )

        if lora_out is not None:
            lora_add = self.mha2sha_optim._op_factory.get_add_op(inp, lora_out)
            self.mha2sha_optim.model.graph.node.append(lora_add)
            inp = lora_add

            self.update_sha_lora_node(
                        branch_type=branch_type,
                        lora_b=lora_out,
                        lora_alpha=lora_inp,  # lora input is alpha
                        lora_add=lora_add
            )

        return inp

    def attach_lora_adaptor(self,
                            info_dict,
                            ns,
                            head_num,
                            head_dim,
                            sha_base_attn_node_list,
                            query_inp,
                            key_inp,
                            value_inp,
                            query_lora_inp,
                            key_lora_inp,
                            value_lora_inp):
        """ Attach lora adpator from lora out to qkv conv """
        query_inp = self.attach_single_lora_adaptor(info_dict["query"],
                                                    ns, head_num, head_dim,
                                                    query_inp, query_lora_inp,
                                                    BranchType.Q)
        key_inp = self.attach_single_lora_adaptor(info_dict["key"],
                                                  ns, head_num, head_dim,
                                                  key_inp, key_lora_inp,
                                                  BranchType.K)
        value_inp = self.attach_single_lora_adaptor(info_dict["value"],
                                                    ns, head_num, head_dim,
                                                    value_inp, value_lora_inp,
                                                    BranchType.V)

        return query_inp, key_inp, value_inp

    def create_sha_conv_lora_rope(self, 
                                  info_dict,
                                  ns,
                                  head_num,
                                  head_dim,
                                  query_matmul_inp,
                                  key_matmul_inp,
                                  value_matmul_inp,
                                  sha_encoding_name_dict,
                                  **extenstion_kwargs):
        return self.mha2sha_optim._mha_conv_extension.create_sha_conv_with_rope(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_encoding_name_dict,
                    **extenstion_kwargs
                )

    def create_sha_conv_lora(self, 
                            info_dict,
                            ns,
                            head_num,
                            head_dim,
                            query_matmul_inp,
                            key_matmul_inp,
                            value_matmul_inp,
                            sha_encoding_name_dict,
                            **extension_kwargs):
        return self.mha2sha_optim.create_sha(
                    info_dict,
                    ns,
                    head_num,
                    head_dim,
                    query_matmul_inp,
                    key_matmul_inp,
                    value_matmul_inp,
                    sha_encoding_name_dict,
                    **extension_kwargs)
