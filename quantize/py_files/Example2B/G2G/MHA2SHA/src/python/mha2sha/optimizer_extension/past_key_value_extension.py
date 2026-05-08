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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from mha2sha.utils.encoding_mapper_utils import (
    create_activation_node_mapping_dict,
    NodeMappingDict,
    update_multi_input_concat_sha_tensor_to_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict,
)

from mha2sha.utils.onnx import (
    NodeNotFoundError,
    get_next_node_down_based_on_cond,
    get_next_node_up_based_on_cond,
)

from onnx import helper
from onnx.onnx_pb import NodeProto, TensorProto

mha2sha_hf_model_optimizer = Any  # Causes circular import


PAST_KEY_VALUE_ACTIVATION_ENCODING_KEYS = [
    "past_key_concat",
    "past_key_concat_out",
    "past_value_concat",
    "past_value_concat_out",
]


@dataclass
class PastKeyValueConcat:
    """Defines nodes to record for past key/values"""

    past_key_concat: Optional[Union[NodeProto, List[NodeProto]]] = None
    past_key_concat_out: Optional[Union[NodeProto, List[NodeProto]]] = None
    past_value_concat: Optional[Union[NodeProto, List[NodeProto]]] = None
    past_value_concat_out: Optional[Union[NodeProto, List[NodeProto]]] = None


def create_past_key_value_encoding_mapping_dict(
    mha_past_kv_dict: Optional[PastKeyValueConcat],
) -> Dict[str, NodeMappingDict]:
    """"""
    if not mha_past_kv_dict:
        return None

    kv = ("key", "value")
    concat_out_dict = {}
    concat_dict = {}

    for attr in kv:
        if concat_out_full_attr := getattr(mha_past_kv_dict, f"past_{attr}_concat_out"):
            # For concat out, they use input and output tensor from same node.output
            concat_out_dict |= {
                f"past_{attr}_concat_out": create_activation_node_mapping_dict(
                    input_1_name=concat_out_full_attr.output[0],
                    output_name=concat_out_full_attr.output[0],
                )
            }

        if concat_full_attr := getattr(mha_past_kv_dict, f"past_{attr}_concat"):
            concat_dict |= {
                f"past_{attr}_concat": create_activation_node_mapping_dict(
                    concat_full_attr.input[0],
                    concat_full_attr.input[1],
                    concat_full_attr.output[0],
                )
            }

    return concat_out_dict | concat_dict


def update_past_key_value_sha_encoding_name_to_encoding_mapping_dict(
    past_key_value_encoding_mapping_dict,
    sha_past_kv_node,
):
    """"""
    for act_name in PAST_KEY_VALUE_ACTIVATION_ENCODING_KEYS:
        if act_name in past_key_value_encoding_mapping_dict:
            node_mapping_dict = past_key_value_encoding_mapping_dict[act_name]
            sha_node = getattr(sha_past_kv_node, act_name)
            if act_name.endswith("concat_out"):
                update_multi_input_concat_sha_tensor_to_node_mapping_dict(
                    node_mapping_dict, sha_node[0]
                )
            else:
                update_sha_tensor_to_node_mapping_dict(node_mapping_dict, sha_node)


class PastKeyValueExtension:
    """Extension for helping with Past Key/Values in LLMs"""

    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info.
        """
        self.mha2sha_optim = mha2sha_optim

    def reset_sha_encoding_name_list(self):
        """
        Reset mha sha names for past key/value tensors.
        """
        self.past_key_value_concat = PastKeyValueConcat(
            past_key_concat=[],
            past_key_concat_out=[],
            past_value_concat=[],
            past_value_concat_out=[],
        )

    def search_past_key_past_value_name(
        self, key_inp, value_inp, qk_matmul, qkv_matmul
    ):
        """
        :param key_inp: input node to key transpose before qk matmul
        :param value_inp: input node to reshape before qkv matmul
        :param qk_matmul: matmul node for qk
        :param qkv_matmul: matmul node for qkv
        :return past_key_output_name: past_key name in the attention
        :return past_value_output_name: past_value name in the attention
        """
        # Search past_key output name between key_matmul and qk_matmul
        past_key_output_name = None
        if self.mha2sha_optim.handle_past_key_value:
            _search_node = key_inp
            while _search_node.name != qk_matmul.name:
                if _search_node.output[0] in self.mha2sha_optim.mha_model_output_names:
                    past_key_output_name = _search_node.output[0]
                    break
                _search_node = self.mha2sha_optim.get_node_by_input_name[
                    _search_node.output[0]
                ][0]
            assert past_key_output_name is not None, "past_key_output_name not found"

        # Search past_value output name between value_matmul and qkv_matmul
        past_value_output_name = None
        if self.mha2sha_optim.handle_past_key_value:
            _search_node = value_inp
            while _search_node.name != qkv_matmul.name:
                if _search_node.output[0] in self.mha2sha_optim.mha_model_output_names:
                    past_value_output_name = _search_node.output[0]
                    break
                _search_node = self.mha2sha_optim.get_node_by_input_name[
                    _search_node.output[0]
                ][0]
            assert (
                past_value_output_name is not None
            ), "past_value_output_name not found"

        return past_key_output_name, past_value_output_name

    def get_kv_cache(self, v_matmul, qkv_matmul, qk_matmul):
        """
        Step 1: Search downstream from v_matmul to qkv_matmul. If there's a concat, then it is kv_cache model.
        Step 2: If it is a kv_cache model, then search upstream from qk_matmul for model_input and model_output.
        """
        kv_cache = False

        # Check if there's a concat between v matmul and qkv matmul for kv_cache mode
        try:
            v_concat = get_next_node_down_based_on_cond(
                v_matmul,
                self.mha2sha_optim.get_node_by_input_name,
                node_found_cond=lambda n: n.op_type == "Concat",
                node_end_search_cond=lambda n: n == qkv_matmul
            )
        except NodeNotFoundError:
            return kv_cache, None, None

        kv_cache = True

        k_input_node = self.mha2sha_optim.get_node_by_output_name[qk_matmul.input[1]]
        if self.mha2sha_optim.gqa_model:
            # Skip Reshape -> expand -> reshape in real_k_input_node(Transpose or Concat) -> Reshape -> expand -> Reshape -> qk_matmul pattern.
            k_input_node = get_next_node_up_based_on_cond(
                self.mha2sha_optim.get_node_by_output_name[k_input_node.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Reshape",
                node_end_search_cond=lambda node: node.op_type
                in ("MatMul", "Conv"),  # End search if we hit a MatMul or Conv
            )
            k_input_node = self.mha2sha_optim.get_node_by_output_name[
                k_input_node.input[0]
            ]

        if (
            k_input_node.input[0]
            in self.mha2sha_optim.mha_model_input_names_index_dict.keys()
        ):
            self.mha2sha_optim.return_new_key_value_only = True
        elif k_input_node.input[0] in self.mha2sha_optim.mha_model_output_names:
            self.mha2sha_optim.return_new_key_value_only = False
        else:
            raise ValueError(
                "Except qk_Matmul.input[1] node's input is model input or model output."
            )

        if self.mha2sha_optim.return_new_key_value_only:
            # k_input_node.input[0] is past_key_in
            k_concat = k_input_node
            k_tranpose = self.mha2sha_optim.get_node_by_output_name[k_concat.input[1]]
        else:
            # k_input_node.input[0] is past_key_out i.e. already concat
            k_tranpose = k_input_node
            k_concat = self.mha2sha_optim.get_node_by_output_name[k_tranpose.input[0]]
        # MATT TODO
        return kv_cache, k_concat, v_concat

    def concat_past_key_value_input(
        self, past_key_inp, past_value_inp, key_inp, value_inp, head_num
    ):
        """
        Slice for given head num on past_key and past_value and concat to current key and value
        :param past_key_inp: [head, 1, past_seq_len, emd_dim] or [1, head, past_seq_len, emd_dim]
                             [head, 1, emd_dim, past_seq_len] to [1, head, emd_dim, past_seq_len]
        :param past_value_inp: [head, 1, past_seq_len, emd_dim] or [1, head, past_seq_len, emd_dim]
        :param key_inp: [1, 1, emd_dim, seq_len]/[B, 1, D, L]
        :param value_inp: [1, 1, seq_len, emd_dim]/[B, 1, L, D]
        :param head_num: head num to slice
        :param slice_dim: dim to slice on past_key and past_value
        :return concated_key_node: [1, 1, emd_dim, seq_len]
        :return concated_value_node: [1, 1, seq_len, emd_dim]
        """
        # check key value head on dim 0 or 1
        past_key_tp = self.mha2sha_optim.model.graph.input[
            self.mha2sha_optim.mha_model_input_names_index_dict[past_key_inp]
        ]
        past_value_tp = self.mha2sha_optim.model.graph.input[
            self.mha2sha_optim.mha_model_input_names_index_dict[past_value_inp]
        ]

        # Swapping Dims 0 and 1 for past keys
        for past_k_v_tp in (past_key_tp, past_value_tp):
            # TODO: Investigate why we have to do it this way?
            # If we just put the swap and not check the name, then the
            # swap happens over and over again.
            # ref: https://github.qualcomm.com/ernst/MHA2SHA/pull/93
            if (
                "past" in past_k_v_tp.name
                and past_k_v_tp.name not in self.mha2sha_optim._seen_past_key_values
            ):
                (
                    past_k_v_tp.type.tensor_type.shape.dim[0].dim_value,
                    past_k_v_tp.type.tensor_type.shape.dim[1].dim_value,
                ) = (
                    past_k_v_tp.type.tensor_type.shape.dim[1].dim_value,
                    past_k_v_tp.type.tensor_type.shape.dim[0].dim_value,
                )
                self.mha2sha_optim._seen_past_key_values.add(past_k_v_tp.name)

        past_key_shape = [d.dim_value for d in past_key_tp.type.tensor_type.shape.dim]
        past_value_shape = [
            d.dim_value for d in past_value_tp.type.tensor_type.shape.dim
        ]
        key_slice_dim = (
            1 if past_key_shape[0] == 1 else 0
        )  # slice on dim=1 if batch dim=1
        value_slice_dim = (
            1 if past_value_shape[0] == 1 else 0
        )  # slice on dim=1 if batch dim=1

        # Check if past key input is transposed
        key_is_transposed = past_key_shape[2] == past_value_shape[2]
        if key_is_transposed:
            #  key_inp: [1, 1, emd_dim, seq_len] and past_key_seq_len: [1, head, past_seq_len, emd_dim]
            #  transpose on past_key_seq_len to: [1, head, emd_dim, past_seq_len]
            past_key_inp = self.mha2sha_optim._op_factory.get_transpose_op(
                past_key_inp, [0, 1, 3, 2]
            )
            self.mha2sha_optim.model.graph.node.append(past_key_inp)

        # Slice from past key and concate with current key
        # key_inp: [1, 1, emd_dim, seq_len]
        past_key_node, past_key_init_list = self.mha2sha_optim._op_factory.get_slice_op(
            past_key_inp, start=head_num, end=head_num + 1, axis=key_slice_dim
        )
        concated_key_node = self.mha2sha_optim._op_factory.get_concat_op(
            [past_key_node, key_inp], 3
        )  # Concate on seq_len
        self.past_key_value_concat.past_key_concat.append(concated_key_node)
        self.mha2sha_optim.model.graph.initializer.extend(past_key_init_list)
        self.mha2sha_optim.model.graph.node.extend([past_key_node, concated_key_node])

        # value_inp: [1, 1, seq_len, emd_dim]
        past_value_node, past_value_init_list = (
            self.mha2sha_optim._op_factory.get_slice_op(
                past_value_inp, start=head_num, end=head_num + 1, axis=value_slice_dim
            )
        )
        concated_value_node = self.mha2sha_optim._op_factory.get_concat_op(
            [past_value_node, value_inp], 2
        )  # Concate on seq_len
        self.past_key_value_concat.past_value_concat.append(concated_value_node)
        self.mha2sha_optim.model.graph.initializer.extend(past_value_init_list)
        self.mha2sha_optim.model.graph.node.extend(
            [past_value_node, concated_value_node]
        )

        if self.mha2sha_optim._ar_builder.buildable:
            self.mha2sha_optim._ar_builder.update_past_key_value_inputs(
                past_key_tp, past_value_tp, key_is_transposed
            )

        return concated_key_node, concated_value_node

    def add_past_key_value_for_llama_sha(
        self,
        past_key_name,
        past_value_name,
        head_num,
        head_dim,
        key_inp_list,
        value_inp_list,
        past_seq_len_input,
        concat_dim=0,
    ):
        """
        delete past_key and past_value from llama grpah.output and past_key and past_value from
        sha split mode to graph.output.
        :param past_key_name: past_key_name to be removed in model.graph.output
        :param past_value_name: past_value_name to be removed in model.graph.output
        :param head_num: number of head
        :param head_dim: head_dim for q, k and v.
        :param key_inp_list: list of tensor with shape [1, 1, emd_dim, seq_len]
        :param value_inp_list: list of tensor with shape [1, 1, seq_len, head_dim]
        :param past_seq_len_input: past key value seq len from kv_cache input
        :param concat_dim: dim to concat heads together.
        :return past_key: shape [..., head_dim, seq_len]
        :return past_value: shape [..., seq_len, head_dim]
        """
        assert concat_dim in (
            0,
            1,
        ), f"Support concat past key and value head in dim 0 or 1, but got {concat_dim}"

        key_value_output_index = []
        for i, _output_tensor in enumerate(self.mha2sha_optim.model.graph.output):
            if _output_tensor.name in [past_key_name, past_value_name]:
                key_value_output_index.append(i)

        for i in sorted(key_value_output_index, reverse=True):
            del self.mha2sha_optim.model.graph.output[i]

        output_seq_len = self.mha2sha_optim.seq_len
        if (
            self.mha2sha_optim.kv_cache
            and not self.mha2sha_optim.return_new_key_value_only
        ):
            output_seq_len = past_seq_len_input + self.mha2sha_optim.seq_len

        # handle past_keys
        # key_inp_list = [1, 1, emd_dim, seq_len]
        key_concat_node = self.mha2sha_optim._op_factory.get_concat_op(
            key_inp_list, concat_dim
        )
        key_concat_node.output[0] = past_key_name
        self.mha2sha_optim.model.graph.node.append(key_concat_node)
        self.past_key_value_concat.past_key_concat_out.append(key_concat_node)

        # Make output tensor
        pask_key_output_shape = [1, 1, int(head_dim), output_seq_len]
        pask_key_output_shape[concat_dim] = int(head_num)
        past_key_output_tensor = helper.make_tensor_value_info(
            past_key_name, TensorProto.FLOAT, pask_key_output_shape
        )

        # handle past_value
        # value_inp_list [1, 1, seq_len, head_dim]
        value_concat_node = self.mha2sha_optim._op_factory.get_concat_op(
            value_inp_list, concat_dim
        )  # head_num*[1, 1, seq_len, head_dim] -> [1, head_num, seq_len, head_dim]
        self.past_key_value_concat.past_value_concat_out.append(value_concat_node)

        if self.mha2sha_optim.nchw_aligned:
            # value_concat_node shape: [head_num, head_dim, 1, seq_len] -> transpose [head_num, 1, seq_len, head_dim]
            value_transpose_node = self.mha2sha_optim._op_factory.get_transpose_op(
                value_concat_node, [0, 2, 3, 1]
            )
            value_transpose_node.output[0] = past_value_name
            self.mha2sha_optim.model.graph.node.extend(
                [value_concat_node, value_transpose_node]
            )
        else:
            value_concat_node.output[0] = past_value_name
            self.mha2sha_optim.model.graph.node.append(value_concat_node)

        past_value_output_shape = [1, 1, output_seq_len, int(head_dim)]
        past_value_output_shape[concat_dim] = int(head_num)
        past_value_output_tensor = helper.make_tensor_value_info(
            past_value_name, TensorProto.FLOAT, past_value_output_shape
        )

        self.mha2sha_optim.model.graph.output.append(past_key_output_tensor)
        self.mha2sha_optim.model.graph.output.append(past_value_output_tensor)
