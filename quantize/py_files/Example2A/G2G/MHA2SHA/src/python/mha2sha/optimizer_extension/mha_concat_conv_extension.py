from .mha_conv_extension import MhaConvExtension

from onnx import numpy_helper
from mha2sha.utils.onnx import get_next_node_up_based_on_cond
from mha2sha.utils.logger import log_debug, log_error, log_info, log_warning
from mha2sha.utils.op_factory import create_tensor_name
from onnx.onnx_pb import NodeProto
from typing import Any, Dict, Tuple, List
from dataclasses import dataclass
from mha2sha.utils.utils import sha_node_name_basis
from mha2sha.utils.utils import BranchType, ExtensionTypes
from mha2sha.utils.encoding_mapper_utils import (
    NodeMappingDict,
    create_activation_node_mapping_dict,
    update_sha_tensor_to_node_mapping_dict
)
CONCAT_CONV_ACTIVATION_ENCODING_KEYS = [
    "conv_1",
    "concat"
]
CONCAT_CONV_PARAM_ENCODING_KEYS = [
    "conv_1_param",
]

@dataclass
class ConcatLinearNode:
    """ Defines nodes to record for lora models. """
    concat: NodeProto = None
    linears: List[NodeProto] = None # in order

    def __str__(self):
        # useful for printing and debuging
        return f"ConcatConvNode(\n" \
               f"    concat: '{self.concat.name if self.concat else None}',\n" \
               f"    linears: '{[x.name for x in self.linears] if self.linears else None}',\n" \
               f")"

    def __repr__(self):
        return str(self)

class MhaConcatConvExtension(MhaConvExtension):

    def reset_sha_encoding_name_list(self):
        self.sha_nodes = {
            t:[]
            for t in [BranchType.Q, BranchType.K, BranchType.V]
        }

    def find_all_conv_before_concat(self, concat_node, conv_op_type):
        ignore_op_types = ["Transpose", "Sequeeze", "Unsequeeze", "Reshape", "Constant", "Identity"]
        all_convs = []
        # find bottom up by BFS
        path_to_check = [concat_node]
        while len(path_to_check) > 0:
            curr_node = path_to_check.pop(0)
            parent_nodes = []
            # parent_nodes should be visited in order (concat inputs have order)
            for t in curr_node.input:
                if t in self.mha2sha_optim.get_node_by_output_name:
                    parent_nodes.append(self.mha2sha_optim.get_node_by_output_name[t])

            for n in parent_nodes:
                if n.op_type in ignore_op_types:
                    path_to_check.append(n)
                elif n.op_type == conv_op_type:
                    all_convs.append(n)
                else:
                    raise ValueError("found unexpected node '{}' in finding all {} of the concat '{}'".\
                                     format(n.name, conv_op_type, concat_node.name))
        return all_convs


    def get_qkv_concat(self, qk_matmul_node, qkv_matmul_node):
        """
        Q = Query, K = Key, V = Value

        Function to search up the the branches of the QK MatMul to find the QK's origin Conv's and search the QKV
        Conv to find V's origin Conv.

        For Stable Diffusion 3's MMDiT, the q/k/v are concated by multiple projections

        +----------+    +----------+      +----------+     +----------+
        | K Conv1  |    |  Q Conv2 |      | K Conv1  |      | K Conv2  |
        +----------+    +----------+      +----------+     +----------+
               \          |                 |           /
                \         |                 |          /
                +-----------+             +-----------+
                | Q Concat  |             | K Concat  |
                +-----------+             +-----------+
                            \               /
                        input 0       input 1
                                +-----------+                 +----------+     +----------+
                                | QK MatMul |                 | V Conv1  |      | V Conv2  |
                                +-----------+                 +----------+     +----------+
                                    |                           |           /
                                    |                           |          /
                                    |                         +-----------+
                                    |                         |  V Concat |
                                intput 0                      +-----------+
                                +------------+                       |
                                | QKV MatMul | -- input 1 ---------- |
                                +------------+

            :param NodeProto qk_matmul_node: The MatMul node where the Q and K branches meet.
            :param NodeProto qkv_matmul_node: The MatMul node where the Q, K, and V branches meet.
            :return List[NodeProto]: The Q, K, and V origin MatMuls.
        """

        q_concat = get_next_node_up_based_on_cond(
            self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[0]],
                self.mha2sha_optim.get_node_by_output_name,
                node_found_cond=lambda n: n.op_type == "Concat"
            )
        print("q_concat",q_concat)
        k_concat = get_next_node_up_based_on_cond(
                    self.mha2sha_optim.get_node_by_output_name[qk_matmul_node.input[1]],
                    self.mha2sha_optim.get_node_by_output_name,
                    node_found_cond=lambda n: n.op_type == "Concat"
                )
        print("k_concat",k_concat)
        v_concat = get_next_node_up_based_on_cond(
            self.mha2sha_optim.get_node_by_output_name[qkv_matmul_node.input[1]],
            self.mha2sha_optim.get_node_by_output_name,
            node_found_cond=lambda n: n.op_type == "Concat"
        )
        print("v_concat",v_concat)
        return [q_concat, k_concat, v_concat]



    def get_dqkv_for_qkv_info(self, concat_node):
        assert concat_node.op_type == "Concat", f"expect node op_type == Concat, but got {concat_node.op_type}"
        all_convs = self.find_all_conv_before_concat(concat_node, "Conv")

        concat_linear_node = ConcatLinearNode(
                                    concat=concat_node,
                                    linears=all_convs
                                )
        # conv_weight_list = [self.get_conv_weight_in_OI(x) for x in all_convs]
        # conv_bias_list = [self.get_conv_bias_in_OI(x) for x in all_convs]

        # - 'matmul_init' and 'matmul_init_bias' should be a string,
        #    so we choose to store the weight/bias of the first conv to them,
        #    and to store all convs information, we should store the rest of them into new field
        return {
                    "matmul_node": all_convs[0],
                    "matmul_init": self.get_conv_weight_in_OI(all_convs[0]),
                    "matmul_init_bias": self.get_conv_bias_in_OI(all_convs[0]),
                    # "other_concat_matmul_init": conv_weight_list[1:],
                    # "other_concat_matmul_init_bias": conv_bias_list[1:],
                    "concat_linear_node":concat_linear_node,
                }

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
            # Still call them matmuls although they are actually convs
            query_concat, key_concat, value_concat = self.get_qkv_concat(qk_matmul_node, qkv_matmul_node)

            if not (query_concat and key_concat and value_concat):
                log_debug("Cannot find QKV concat.")
            else:
                log_debug(
                    f"QKV concat : {query_concat.name}, {key_concat.name}, {value_concat.name}"
                )
            dquery = self.get_dqkv_for_qkv_info(query_concat)
            dkey = self.get_dqkv_for_qkv_info(key_concat)
            dvalue = self.get_dqkv_for_qkv_info(value_concat)


            return dquery, dkey, dvalue


    def create_single_branch(self,
                           qkv_init,
                           ns,
                           head_num,
                           head_dim,
                           suffix=None,
                           branch_type=None,
                           bias_init=None,
                           info=None,
                           ):

        sha_convs = []
        sha_name_basis = sha_node_name_basis(ns, head_num)[branch_type.value]
        for i,mha_conv in enumerate(info["concat_linear_node"].linears):

            sha_qkv_name = sha_name_basis + "_{}".format(i)
            sha_weight_name = sha_qkv_name+".weight" if suffix is None else sha_qkv_name+f"_{suffix}.weight"
            sha_bias_name = sha_qkv_name+".bias" if suffix is None else sha_qkv_name+f"_{suffix}.bias"

            # Create weight names
            conv_init_name, self.mha2sha_optim.tensor_name_set = create_tensor_name(
                sha_weight_name, self.mha2sha_optim.tensor_name_set
            )
            assert sha_weight_name == conv_init_name, f"sha_query_weight_name and created init name are differnet"
            bias_init_name, self.mha2sha_optim.tensor_name_set = create_tensor_name(
                sha_bias_name, self.mha2sha_optim.tensor_name_set
            )
            # Create q, k, v conv weight
            # info_dict["query"]["matmul_init"] has shape [I, O]
            # Conv weight has shape [head_dim, I, kH, kW]
            conv_init = numpy_helper.from_array(
                self.get_conv_weight_in_OI(mha_conv)[
                    :, head_num * head_dim : (head_num + 1) * head_dim
                    ].T[..., None, None],
                conv_init_name,
            )

            # create bias if exists
            if bias_init is not None:
                bias_init = numpy_helper.from_array(
                                    self.get_conv_bias_in_OI(mha_conv)[
                                        head_num * head_dim : (head_num + 1) * head_dim],
                                    bias_init_name,
                            )
            else:
                bias_init_name = None

            # Create q, k, v matmul op
            conv_node = self.mha2sha_optim._op_factory.get_conv_op(
                            input_node=mha_conv.input[0],
                            weight_tensor_name=conv_init_name,
                            bias_tensor_name=bias_init_name,
                            kernel_shape=[1, 1],
                            padding=[0, 0, 0, 0],
                            strides=[1, 1],
                            propose_op_name=sha_qkv_name+"_Conv" if suffix is None else sha_qkv_name+f"_{suffix}_Conv",
                            output_tensor_name=None
                        )
            self.mha2sha_optim.model.graph.initializer.append(conv_init)
            if bias_init is not None:
                self.mha2sha_optim.model.graph.initializer.append(bias_init)
            self.mha2sha_optim.model.graph.node.append(conv_node)

            sha_convs.append(conv_node)

        concat_node = self.mha2sha_optim._op_factory.get_concat_op(
            [x.output[0] for x in sha_convs], axis=-1, namehint="{}_Concat".format(sha_name_basis)
        )

        self.mha2sha_optim.model.graph.node.append(concat_node)

        sha_concat_conv_node = ConcatLinearNode(
            concat=concat_node,
            linears=sha_convs
        )

        self.sha_nodes[branch_type].append(sha_concat_conv_node)

        return concat_node, sha_convs[0] # the first conv can be handled by original pipeline

    def create_sha_qkv_convs(self,
                             info_dict,
                             ns,
                             head_num,
                             head_dim,
                             sha_base_attn_node_list,
                             query_matmul_inp,
                             key_matmul_inp,
                             value_matmul_inp,
                             init_name="matmul_init",
                             suffix=None,
                             ):
        """
        Create MatMuls with initializer sliced by head.
        :return query_inp: query linear node
        :return key_inp: key linear node
        :return value_inp: value linear node
        """
        _, propose_sha_query_name, propose_sha_key_name, propose_sha_value_name = sha_node_name_basis(ns, head_num)

        # Create weight names
        query_first_conv = None
        key_first_conv = None
        value_first_conv = None
        if init_name in info_dict["query"].keys() and (conv_weight_init := info_dict["query"][init_name]) is not None:
            query_end_node, query_first_conv = self.create_single_branch(
                                                    conv_weight_init,
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    suffix=suffix,
                                                    branch_type=BranchType.Q,
                                                    bias_init=info_dict["query"].get(init_name+"_bias", None),
                                                    info=info_dict["query"],
                                                )

        if init_name in info_dict["key"].keys() and (conv_weight_init := info_dict["key"][init_name]) is not None:
            key_end_node, key_first_conv = self.create_single_branch(
                                                conv_weight_init,
                                                ns,
                                                head_num,
                                                head_dim,
                                                suffix=suffix,
                                                branch_type=BranchType.K,
                                                bias_init=info_dict["key"].get(init_name+"_bias", None),
                                                info=info_dict["key"],
                                            )

        if init_name in info_dict["value"].keys() and (conv_weight_init := info_dict["value"][init_name]) is not None:
            value_end_node, value_first_conv = self.create_single_branch(
                                                    conv_weight_init,
                                                    ns,
                                                    head_num,
                                                    head_dim,
                                                    suffix=suffix,
                                                    branch_type=BranchType.V,
                                                    bias_init=info_dict["value"].get(init_name+"_bias", None),
                                                    info=info_dict["value"],
                                                )

        if suffix is None:
            sha_base_attn_node_list.q_matmul.append(query_first_conv)
            sha_base_attn_node_list.k_matmul.append(key_first_conv)
            sha_base_attn_node_list.v_matmul.append(value_first_conv)

        if suffix is None and (query_end_node or key_end_node or value_end_node) is None:
            raise ValueError(f"Can not find Q, K, or V got: {query_end_node, key_end_node, value_end_node}")

        query_inp = query_end_node
        key_inp = key_end_node
        value_inp = value_end_node

        return query_inp, key_inp, value_inp

def update_concat_projections_encoding_name_to_rest_encoding_mapping_dict(
        rest_concat_projections,
        sha_nodes,
        mha_info_dict,
    ):
    """
    Update the rest sha projections encoding names to EncodingMappingDict.

    The first projection should have been handled by update_base_attn_sha_encoding_name_to_base_attn_encoding_mapping_dict
    Thus, in this function, we only need to handle the rest of them

    """
    prefix = ExtensionTypes.CONCAT_CONV.value
    for branch_type, branch_type_str in [(BranchType.K, "key"), (BranchType.Q, "query"), (BranchType.V, "value")]:
        mha_concat_linear_node = mha_info_dict[branch_type_str]["concat_linear_node"]
        branch_type_alias_str = branch_type_str[0]
        for conv_i, mha_conv in enumerate(mha_concat_linear_node.linears[1:]):
            conv_i += 1
            conv_i_node_mapping = create_activation_node_mapping_dict(
                                        input_1_name=mha_conv.input[0],
                                        output_name=mha_conv.output[0]
                                    )
            update_sha_tensor_to_node_mapping_dict(
                conv_i_node_mapping,
                [x.linears[conv_i] for x in sha_nodes[branch_type]]
            )
            conv_name = "{}_{}_conv_{}".format(prefix, branch_type_alias_str, conv_i)
            rest_concat_projections[conv_name] = conv_i_node_mapping

            rest_concat_projections[conv_name + "_param"] = NodeMappingDict(
                    mha_mapping_name_list = ["mha_param_name"],
                    sha_mapping_name_list = ["sha_param_name"],
                    mapping_name_dict = {
                        "mha_param_name":  mha_conv.input[1],
                        "sha_param_name": [x.linears[conv_i].input[1] for x in sha_nodes[branch_type]]
                    }
            )

        if len(mha_concat_linear_node.linears) > 2:
            # this is because the encodings mapping infrastructure only support 2 inputs for one op for now
            raise ValueError("currently we only support 2 linears concatenation")
        mha_concat = mha_concat_linear_node.concat
        concat_node_mapping = create_activation_node_mapping_dict(
                                input_1_name=mha_concat.input[0],
                                input_2_name=mha_concat.input[1],
                                output_name=mha_concat.output[0]
                            )
        update_sha_tensor_to_node_mapping_dict(
            concat_node_mapping,
            [x.concat for x in sha_nodes[branch_type]]
        )
        rest_concat_projections["{}_{}_concat".format(prefix, branch_type_alias_str)] = concat_node_mapping
