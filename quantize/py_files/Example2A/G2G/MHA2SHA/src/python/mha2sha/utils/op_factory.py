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
"""Factory functions for creating ONNX ops.

This module holds a class to create ONNX ops for a model. The `OpFactory` class takes in aspects of the already
created model, and use this info to provide new ops to add to the model.

Basic usage
-----------

>>> op_factory = OpFactory(
        tensor_name_set,
        model,
        node_name_mapping_dict,
        mha_model_input_names_index_dict,
        model_opset
    )
>>> slice_op = op_factory.get_slice_op(input_node, start=0, end=head_dim//2, axis=3)

"""
from functools import wraps
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from mha2sha.transformations.ar_builder import ArBuilder
import numpy as np
import onnx
from onnx.onnx_pb import AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto
from onnx import helper


def _track_reshape_in_ar_builder(func: Callable):
    r"""Hook for tracking reshape ops in AR Builder.

    Tracking Reshape op's created in the attention modules for updating AR Builder.

    Args:
        'get_reshape_op' function.

    Returns:
        Reshape op and init.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        reshape_op, reshape_init = func(self, *args, **kwargs)
        if self._ar_builder.buildable:
            self._ar_builder.reshapes_not_to_update_for_ar.add(reshape_op.name)
        return reshape_op, reshape_init

    return wrapper


def create_node_name(
    graph: GraphProto,
    op_type: str,
    _node_name_suffix: Dict[str, int],
    name_prefix: str = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Create a unique node name that starts with a prefix (default is operator type).
    The name will not be duplicated with any name that generated or existed in current graphs.

    :param graph (GraphProto): Onnx GraphProto instance of model
    :param op_type (str): Operator type for which the name is to be generated.
    :param _node_name_suffix (Dict[str, int]): Dict mapping of node_name to its suffix.
    :param name_prefix (str, optional): Prefix of node name. Defaults to None.
    :returns Tuple[str, Dict[str, int]]: Node name for given node op type and updated
                                         Dict mapping of node_name to its suffix.
    """
    # TODO: This functionality shall be redesigned where node can be created
    #        first and then added to graph and post that we shall call an API
    #        assign_name on graph to address issues related to empty name.
    if name_prefix:
        prefix = name_prefix if name_prefix.endswith("_") else (name_prefix + "_")
    else:
        prefix = op_type + "_"
    suffix: int = 0
    if prefix in _node_name_suffix:
        suffix = _node_name_suffix[prefix] + 1
    else:
        # Check existed node name only once for a prefix as we assume
        # create_node_name is called for every new node in fusion.
        for node in graph.node:
            if node.name and node.name.startswith(prefix):
                try:
                    index = int(node.name[len(prefix):])
                    suffix = max(index + 1, suffix)
                except ValueError:
                    continue
    # Record the generated suffix so that we can avoid generating duplicated name.
    _node_name_suffix[prefix] = suffix
    return prefix + str(suffix), _node_name_suffix


def create_tensor_name(
    proposed_tensor_name: str, tensor_name_set: Set[str]
) -> Tuple[str, Set[str]]:
    """
    Function to create a new tensor name which doesnt conflict with existing
    tensor names.

    :param  proposed_tensor_name (str): Proposed name of the new tensor.
    :param tensor_name_set (Set[str]): Set of output tensor names of the model.
    :returns Tuple[str, Set[str]]: Tuple of updated name of the new tensor and
        updated set of the output tensor names of the model.
    """
    new_name = proposed_tensor_name
    counter = 1
    while new_name in tensor_name_set:
        new_name = f"{proposed_tensor_name.split('_')[0]}_{counter}"
        counter += 1
    tensor_name_set.add(new_name)
    return new_name, tensor_name_set


def get_opset_version(model: ModelProto) -> int:
    """
    Return the model opset version for default domain.

    :param ModelProto model: Onnx model proto instance.
    :raises RuntimeError: If no default domains found in model.
    :return int: opset version of onnx domain
    """
    for opset in model.opset_import:
        if opset.domain in ["", "ai.onnx"]:
            return opset.version
    raise RuntimeError("Onnx model has no opset for default domain")


# OpFactory helper functions
def make_tensor(name: str, data: Union[np.ndarray, List]) -> TensorProto:
    """
    Function to generate TensorProto object based on given datatype, dims and values.

    :param str name: Name of the TensorProto.
    :param Union[np.ndarray, List] data: Actual data to be used for the TensorProto.
    :return TensorProto: return tensor proto.
    """
    if isinstance(data, List):
        data = np.array(data, dtype=np.float32)

    tensor = helper.make_tensor(
        name=name,
        data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype],
        dims=data.shape,
        vals=data.flatten().tolist(),
    )
    return tensor


def make_node(
    op_type: str,
    inputs: List[str],
    outputs: List[str],
    name: Optional[str] = None,
    doc_string: Optional[str] = None,
    domain: Optional[str] = None,
    **kwargs: Dict,
) -> NodeProto:
    """
    Function to generate node based on given params and doc_string

    :param str op_type: Node operator type
    :param List[str] inputs: List of input node names
    :param List[str] outputs: List of output node names
    :param Optional[str] name: Name of the node. Defaults to None.
    :param Optional[str] doc_string: Doc string used to describe the graph.
        Defaults to None.
    :param Optional[str] domain: Domain name for the node. Defaults to None.
    :return NodeProto: NodeProto of the generated Node
    """
    node = helper.make_node(
        op_type, inputs, outputs, name, doc_string, domain, **kwargs
    )
    if doc_string == "":
        node.doc_string = ""
    order_repeated_field(node.attribute, "name", kwargs.keys())
    return node


def order_repeated_field(
    repeated_proto: AttributeProto, key_name: str, order: List[str]
) -> None:
    """
    Function to sort the fields in NodeProto.

    :param AttributeProto repeated_proto: NodeProto of a node
    :param str key_name: key_name for each attribute
    :param List[str] order: List of arguments for a node
    """
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))


class OpFactory:
    """Factory class for creating ONNX ops.

    This class is designed to abstract the creation of ONNX ops. Each function is noted as `get_` followed by
    the name of the op to create. For orginization, the functions are in alphabetical order.

    Attributes:

    """
    def __init__(
        self,
        tensor_name_set: Set[str],
        model: ModelProto,
        node_name_mapping_dict: Dict[str, int],
        mha_model_input_names_index_dict: Dict[str, int],
        ar_builder: ArBuilder,
    ) -> None:
        """Initializes the instance based on attributes passed in for the current model.

        Args:
            tensor_name_set:
                Set of unique tensor names.
            model:
                The model loader of the current model.
            node_name_mapping_dict:
                The dictionary mapping node names to number.
            mha_model_input_names_index_dict:
                Dictionary mapping the input names to the index they appear.
        """

        self._tensor_name_set = tensor_name_set
        self._model = model
        self._node_name_mapping_dict = node_name_mapping_dict
        self._mha_model_input_names_index_dict = mha_model_input_names_index_dict
        self._ar_builder = ar_builder
        self._opset_version = get_opset_version(model)

    def get_sub_op(self, input_node_1: NodeProto, input_node_2: NodeProto) -> Tuple[NodeProto, List]:
        """
        Function responsible for creating a Sub operation ONNX node.

        :param input_node_1: NodeProto
        :param input_node_2: NodeProto

        :return sub_node: NodeProto - ONNX node for Sub
        :return []: List - list of initializers required for this opeartion.
        """
        output, self._tensor_name_set = create_tensor_name("Sub", self._tensor_name_set)
        node_name, self.node_name_mapping_dict = create_node_name(
            self._model.graph, "Sub", self.node_name_mapping_dict
        )
        inp1 = (
            input_node_1.output[0]
            if isinstance(input_node_1, NodeProto)
            else input_node_1
        )
        inp2 = (
            input_node_2.output[0]
            if isinstance(input_node_2, NodeProto)
            else input_node_2
        )
        add_node = make_node(
            "Sub",
            inputs=[inp1, inp2],
            outputs=[output],
            name=node_name,
        )
        return add_node

    def get_add_op(self, input_node_1: NodeProto, input_node_2: NodeProto) -> NodeProto:
        """Creates an ONNX Add operation Node.

        Function responsible for creating a Add operation ONNX node.
        
        Args:
            input_node_1:
                First input node into the new Add node.
            input_node_2:
                Second input node into the new Add node.
                
        Returns:
            The newly created ONNX Add node.
        """
        output, self._tensor_name_set = create_tensor_name("Add", self._tensor_name_set)
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Add", self._node_name_mapping_dict
        )
        inp1 = (
            input_node_1.output[0]
            if isinstance(input_node_1, NodeProto)
            else input_node_1
        )
        inp2 = (
            input_node_2.output[0]
            if isinstance(input_node_2, NodeProto)
            else input_node_2
        )
        add_node = make_node(
            "Add",
            inputs=[inp1, inp2],
            outputs=[output],
            name=node_name,
        )
        return add_node
            
    def get_cast_op(self, input_node: NodeProto, to: int) -> NodeProto:
        """Creates an ONNX Cast operation Node.

        Function responsible for creating a Cast operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Cast node.
            to:
                Data type to cast to. Based on the integer Enum TensorProto.Types.DataType

        Returns:
            The newly created ONNX Cast node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        output, self._tensor_name_set = create_tensor_name(
            "Cast", self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Cast", self._node_name_mapping_dict
        )
        cast_node = make_node(
            "Cast",
            inputs=[inp],
            outputs=[output],
            name=node_name,
            to=to,
        )
        return cast_node

    def get_concat_op(self, list_of_input_nodes: List[NodeProto], axis: int, namehint="Concat") -> NodeProto:
        """Creates an ONNX Concat operation Node.

        Function responsible for creating a Concat operation ONNX node.
        
        Args:
            list_of_input_nodes:
                List of input nodes into the new Concat node.
            axis:
                Axis to concatenate along.
            namehint:
                a name hint for the op's name

        Returns:
            The newly created ONNX Concat node.
        """
        
        output, self._tensor_name_set = create_tensor_name(
            namehint, self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, namehint, self._node_name_mapping_dict
        )
        input_to_concat_node = []
        for node in list_of_input_nodes:
            if isinstance(node, NodeProto):
                input_to_concat_node.append(node.output[0])
            elif isinstance(node, str):
                input_to_concat_node.append(node)
            else:
                raise ValueError(f"type of {node} is invalid.")
    
        concat_node = make_node(
            "Concat",
            inputs=input_to_concat_node,
            outputs=[output],
            name=node_name,
            axis=axis,
        )
        return concat_node

    def get_conv_op(
        self,
        input_node: NodeProto,
        weight_tensor_name: str,
        bias_tensor_name: Optional[str] = None,
        kernel_shape: Union[int, List[int]] = 3,
        padding: Union[int, List[int]] = 1,
        strides: Union[int, List[int]] = 1,
        propose_op_name: str = "conv",
        output_tensor_name: Optional[str] = None,
    ) -> NodeProto:
        """Creates an ONNX Conv node.
            
            Creates a Convolution node with the provided inputs, outputs, and attributes of the conv.

            Args:
                input_node:
                    Node that inputs into the new Conv op.
                weight_tensor_name:
                    Name of the weight tensor into the new Conv op. Must be apart of the initializers.
                bias:
                    Name of the bias tensor into the new Conv op. Must be apart of the initializers.
                kernel_shape:
                    Kernel shape, default 3.
                padding:
                   Padding, default 1. 
                strides:
                    Strides, default 1.
                propose_op_name:
                    Prefix name for the weight, bias, and output. Default is "conv". If a name is given, we are using
                    this name as the node of a node to replace. For example, replacing Linear with Conv.
                output_tensor_name:
                    Optional name of the output tensor, if None is given, one will be made.

            Return:
                The newly created ONNX Conv node.
        """

        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        input_list = [inp, weight_tensor_name]
        if bias_tensor_name:
            input_list.append(bias_tensor_name)

        if not output_tensor_name:
            output_tensor_name, self._tensor_name_set = create_tensor_name(
                propose_op_name+"/output", self._tensor_name_set
            )

        # Create and add the node to the node_name_mapping_dict if the propose_op_name isn't the default. This means
        # we are replacing an op, rather than adding a new one.
        if propose_op_name == "conv":
            node_name, self._node_name_mapping_dict = create_node_name(
                self._model.graph, propose_op_name, self._node_name_mapping_dict
            )
        else:
            node_name = propose_op_name

        return make_node(
            "Conv",
            inputs=input_list,
            outputs=[output_tensor_name],
            name=node_name,
            kernel_shape=kernel_shape,
            pads=padding,
            strides=strides
        )

    def get_div_op(self, input_node: NodeProto, value: float, propose_op_name: str = "Div") -> Tuple[NodeProto, List]:
        """Creates an ONNX Div operation Node.

        Function responsible for creating a Div operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Div node.
            value:
                Value to divide by, in floating point.
            propose_op_name:
                The proposed name of the Div node, default is "Div".
                
        Returns:
            A tuple containing the newly created ONNX Div node and a list of initializers required for this node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        div_tensor_name, self._tensor_name_set = create_tensor_name(
            propose_op_name+"_value", self._tensor_name_set
        )
        div_init = make_tensor(
            name=div_tensor_name,
            data=np.array([value]).flatten().astype(np.float32),
        )
        output, self._tensor_name_set = create_tensor_name(propose_op_name+"/output", self._tensor_name_set)
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )
        div_node = make_node(
            "Div",
            inputs=[inp, div_tensor_name],
            outputs=[output],
            name=node_name,
        )
        return div_node, [div_init]

    def get_element_mul_op(self, input_node_1: NodeProto, input_node_2: NodeProto) -> NodeProto:
        """Creates an ONNX Elementwise Mul operation Node.

        Function responsible for creating a Elementwise Mul operation ONNX node.
        
        Args:
            input_node_1:
                First input node into the new Elementwise Mul node.
            input_node_2:
                Second input node into the new Elementwise Mul node.
                
        Returns:
            The newly created ONNX Elementwise Mul node.
        """

        output, self._tensor_name_set = create_tensor_name("Mul", self._tensor_name_set)
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Mul", self._node_name_mapping_dict
        )

        inp1 = (
            input_node_1.output[0]
            if isinstance(input_node_1, NodeProto)
            else input_node_1
        )
        inp2 = (
            input_node_2.output[0]
            if isinstance(input_node_2, NodeProto)
            else input_node_2
        )
        add_node = make_node(
            "Mul",
            inputs=[inp1, inp2],
            outputs=[output],
            name=node_name,
        )

        return add_node


    def get_layer_norm_op(
        self,
        input_node: Union[str, NodeProto],
        scale: Union[str, np.ndarray],
        bias: Optional[Union[str, np.ndarray]] = None,
        axis: Optional[int] = -1,  # default based off documentations
        epsilon: Optional[float] = 1e-05,  # default based off documentations
        stash_type: Optional[int] = 1,  # default based off documentations
        propose_op_name: Optional[str] = "LayerNorm",
        output_tensor_names: Optional[List[str]] = None,
    ) -> Tuple[NodeProto, List]:
        """Creates an ONNX LayerNormalization Op.

        Args:
            input_node:
                Node that inputs into the LayerNorm.
            scale_input:
                Scale input.
            bias:
                Optional bias input.
            axis:
                Optional axis attribute. Defaults to -1 based of LayerNorm ONNX definition.
            epsilon:
                Optional epsilon attribute. Defaults to 1e-05 based of LayerNorm ONNX definition.
            stash_type:
                Optional stash_type attribute. Defaults to 1 based of LayerNorm ONNX definition.
            propose_op_name:
                What to start name of op with.
            output_tensor_names:
                Output tensor names that are already apart of the graph. Otherwise, a new output will be created.

        Returns:
            Created LayerNorm op and initializer.
        """

        input_tensor_name = (
            input_node.output[0]
            if isinstance(input_node, NodeProto) else
            input_node
        )

        if isinstance(scale, np.ndarray):
            scale_tensor_name, self._tensor_name_set = create_tensor_name(
                propose_op_name + "_scale", self._tensor_name_set
            )
            scale_tensor_init = make_tensor(
                name=scale_tensor_name,
                data=scale,
            )
        else:
            scale_tensor_name = scale
            scale_tensor_init = None

        bias_tensor_init = None
        if bias is not None:
            if isinstance(bias, np.ndarray):
                bias_tensor_name, self._tensor_name_set = create_tensor_name(
                    propose_op_name + "_bias", self._tensor_name_set
                )
                bias_tensor_init = make_tensor(
                    name=bias_tensor_name,
                    data=bias,
                )
            else:
                bias_tensor_name = bias

        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )
        if not output_tensor_names:
            output, self._tensor_name_set = create_tensor_name(
                propose_op_name + "/output", self._tensor_name_set
            )
            output_tensor_names = [output]

        layer_norm_node = make_node(
            "LayerNormalization",
            inputs=[input_tensor_name, scale_tensor_name, bias_tensor_name],
            outputs=output_tensor_names,
            axis=axis,
            epsilon=epsilon,
            stash_type=stash_type,
            name=node_name,
        )

        return layer_norm_node, list(filter(None, [scale_tensor_init, bias_tensor_init]))


    def get_matmul_op(
        self,
        input_node_1: NodeProto,
        input_node_2: NodeProto,
        propose_op_name: str = "MatMul" 
    ) -> Tuple[NodeProto, List]:
        """Creates an ONNX MatMul operation Node.

        Function responsible for creating a MatMul operation ONNX node.
        
        Args:
            input_node_1:
                First input node into the new MatMul node.
            Input_node_2:
                Second input node into the new MatMul node.
            propose_op_name:
                The proposed name of the MatMul node, default is "MatMul".
                
        Returns:
            A tuple containing the newly created ONNX MatMul node and a list of initializers required for this node.
        """

        output, self._tensor_name_set = create_tensor_name(
            propose_op_name+"/output", self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )
        inp1 = (
            input_node_1.output[0]
            if isinstance(input_node_1, NodeProto)
            else input_node_1
        )
        inp2 = (
            input_node_2.output[0]
            if isinstance(input_node_2, NodeProto)
            else input_node_2
        )
        matmul_node = make_node(
            "MatMul",
            inputs=[inp1, inp2],
            outputs=[output],
            name=node_name,
        )
        return matmul_node, []

    def get_mul_op(self, input_node: NodeProto, value: float, propose_op_name: str = "Mul") -> Tuple[NodeProto, List]:
        """Creates an ONNX Mul operation Node.

        Function responsible for creating a Mul operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Mul node.
            value:
                Value to multiply by, in floating point.
            propose_op_name:
                The proposed name of the Mul node, default is "Mul".
                
        Returns:
            A tuple containing the newly created ONNX Mul node and a list of initializers required for this node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        mul_tensor_name, self._tensor_name_set = create_tensor_name(
            propose_op_name+"_value", self._tensor_name_set
        )
        mul_init = make_tensor(
            name=mul_tensor_name,
            data=np.array([value]).flatten().astype(np.float32),
        )
        output, self._tensor_name_set = create_tensor_name(propose_op_name+"/output", self._tensor_name_set)
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )
        mul_node = make_node(
            "Mul",
            inputs=[inp, mul_tensor_name],
            outputs=[output],
            name=node_name,
        )
        return mul_node, [mul_init]

    def get_neg_op(self, input_node: NodeProto) -> NodeProto:
        """Creates an ONNX Neg operation Node.

        Function responsible for creating a Neg operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Neg node.
                
        Returns:
            The newly created ONNX Neg node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        output, self._tensor_name_set = create_tensor_name(
            "Neg", self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Neg", self._node_name_mapping_dict
        )

        neg_node = make_node(
            "Neg",
            inputs=[inp],
            outputs=[output],
            name=node_name,
        )
        return neg_node

    @_track_reshape_in_ar_builder
    def get_reshape_op(self, input_node: NodeProto, shape: List[int], output: bool = None) -> Tuple[NodeProto, List]:
        """Creates an ONNX Reshape operation Node.

        Function responsible for creating a Reshape operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Reshape node.
            shape:
                Shape to reshape into.
        
        Returns:
            A tuple containing the newly created ONNX Reshape node and a list of initializers required for this node.
        """

        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )

        reshape_tensor_name, self._tensor_name_set = create_tensor_name(
            "Reshape_tensor", self._tensor_name_set
        )
        reshape_init = make_tensor(
            name=reshape_tensor_name,
            data=np.array(shape).flatten().astype(int),
        )
        
        if output is None:
            output, self._tensor_name_set = create_tensor_name("Reshape", self._tensor_name_set)

        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Reshape", self._node_name_mapping_dict
        )

        reshape_node = make_node(
            "Reshape",
            inputs=[inp, reshape_tensor_name],
            outputs=[output],
            name=node_name,
        )
        return reshape_node, [reshape_init]

    def get_slice_op(self, input_node: NodeProto, start: int, end: int, axis: int) -> Tuple[NodeProto, List]:
        """Creates an ONNX Slice operation Node.

        Function responsible for creating a Slice operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Slice node.
            start:
                Where to start slicing.
            end:
                Where to end slicing.
            axis:
                What axis to slice.
        
        Returns:
            A tuple containing the newly created ONNX Slice node and a list of initializers required for this node.
        """

        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        start_init_tensor, self._tensor_name_set = create_tensor_name(
            "start_init_", self._tensor_name_set
        )
        start_init = make_tensor(
            name=start_init_tensor,
            data=np.array([start]).flatten().astype(int),
        )
        end_init_tensor, self._tensor_name_set = create_tensor_name(
            "end_init_", self._tensor_name_set
        )
        end_init = make_tensor(
            name=end_init_tensor,
            data=np.array([end]).flatten().astype(int),
        )
        axes_init_tensor, self._tensor_name_set = create_tensor_name(
            "axes_init_", self._tensor_name_set
        )
        axes_init = make_tensor(
            name=axes_init_tensor,
            data=np.array([axis]).flatten().astype(int),
        )
        steps_init_tensor, self._tensor_name_set = create_tensor_name(
            "steps_init_", self._tensor_name_set
        )
        steps_init = make_tensor(
            name=steps_init_tensor,
            data=np.array([1]).flatten().astype(int),
        )
        output, self._tensor_name_set = create_tensor_name("Slice", self._tensor_name_set)
        node_name, self.node_name_mapping_dict = create_node_name(
            self._model.graph, "Slice", self._node_name_mapping_dict
        )
        slice_node = make_node(
            "Slice",
            inputs=[
                inp,
                start_init_tensor,
                end_init_tensor,
                axes_init_tensor,
                steps_init_tensor,
            ],
            outputs=[output],
            name=node_name,
        )
        return slice_node, [start_init, end_init, axes_init, steps_init]

    def get_softmax_op(
        self,
        input_node: NodeProto,
        axis: int,
        propose_op_name: str = "Softmax"
    ) -> NodeProto:
        """Creates an ONNX Softmax operation Node.

        Function responsible for creating a Softmax operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Softmax node.
            axis:
                Axis to apply softmax.
            propose_op_name:
                Proposed name for the new op, default is "Softmax".

        Returns:
            The newly created ONNX Softmax node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        output, self._tensor_name_set = create_tensor_name(
            propose_op_name+'/output', self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )
        softmax_node = make_node(
            "Softmax",
            inputs=[inp],
            outputs=[output],
            name=node_name,
            axis=axis,
        )
        return softmax_node

    def get_squeeze_op(self, input_node: NodeProto, axis: int, output: Optional[str] = None) -> Tuple[NodeProto, List]:
        """Creates an ONNX Squeeze operation Node.

        Function responsible for creating a Squeeze operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Squeeze node.
            axis:
                Axis to squeeze on.
            output:
               Optional output of the squeeze op, otherwise one is created. 

        Returns:
            A tuple containing the newly created ONNX Squeeze node and a list of initializers required for this node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        if not output:
            output, self._tensor_name_set = create_tensor_name(
                "Squeeze", self._tensor_name_set
            )

        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Squeeze", self._node_name_mapping_dict
        )
        if self._opset_version >= 13:
            axes_init_tensor, self._tensor_name_set = create_tensor_name(
                "squeeze_axes_init_", self._tensor_name_set
            )
            axes_init = make_tensor(
                name=axes_init_tensor,
                data=np.array([axis]).flatten().astype(int),
            )
            squeeze_node = make_node(
                "Squeeze",
                inputs=[inp, axes_init_tensor],
                outputs=[output],
                name=node_name,
            )
            init_list = [axes_init]
        else:
            squeeze_node = make_node(
                "Squeeze",
                inputs=[inp],
                outputs=[output],
                name=node_name,
                axes=[axis],
            )
            init_list = []
        return squeeze_node, init_list

    def get_transpose_op(
        self,
        input_node: NodeProto,
        perm: List[int],
        propose_op_name: str = "Transpose",
        output: Optional[str] = None
    ) -> NodeProto:
        """Creates an ONNX Transpose operation Node.

        Function responsible for creating a Transpose operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Transpose node.
            perm:
                List for permuting the axes.
            propose_op_name:

            output:
                Used when the output for the Transpose op's output is apart of the model output.

        Returns:
            The newly created ONNX Transpose node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )

        if not output:
            output, self._tensor_name_set = create_tensor_name(
                propose_op_name+"/output", self._tensor_name_set
            )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, propose_op_name, self._node_name_mapping_dict
        )

        transpose_node = make_node(
            "Transpose",
            inputs=[inp],
            outputs=[output],
            name=node_name,
            perm=perm,
        )
        return transpose_node

    def get_unsqueeze_op(self, input_node: NodeProto, axis: int) -> Tuple[NodeProto, List]:
        """Creates an ONNX Unsqueeze operation Node.

        Function responsible for creating a Unsqueeze operation ONNX node.
        
        Args:
            input_node:
                Input node into the new Unsqueeze node.
            axis:
                Axis in which to insert the singleton dimension.

        Returns:
            A tuple containing the newly created ONNX Unsqueeze node and a list of initializers required for this node.
        """
        inp = (
            input_node.output[0]
            if isinstance(input_node, NodeProto)
            else input_node
        )
        output, self._tensor_name_set = create_tensor_name(
            "Unsqueeze", self._tensor_name_set
        )
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Unsqueeze", self._node_name_mapping_dict
        )
        if self._opset_version >= 13:
            axes_init_tensor, self._tensor_name_set = create_tensor_name(
                "unsqueeze_axes_init_", self._tensor_name_set
            )
            axes_init = make_tensor(
                name=axes_init_tensor,
                data=np.array([axis]).flatten().astype(int),
            )
            unsqueeze_node = make_node(
                "Unsqueeze",
                inputs=[inp, axes_init_tensor],
                outputs=[output],
                name=node_name,
            )
            init_list = [axes_init]
        else:
            unsqueeze_node = make_node(
                "Unsqueeze",
                inputs=[inp],
                outputs=[output],
                name=node_name,
                axes=[axis],
            )
            init_list = []
        return unsqueeze_node, init_list

    def get_where_op(self, input_node_1: NodeProto, input_node_2: NodeProto, input_node_3: NodeProto) -> NodeProto:
        """Creates an ONNX Where operation Node.

        Function responsible for creating a Where operation ONNX node.
        
        Args:
            input_node_1:
                First input node into the new Where node.
            input_node_2:
                Second input node into the new Where node.
            input_node_3:
                Third input node into the new Where node.

        Returns:
            The newly created ONNX Where node.
        """

        output, self._tensor_name_set = create_tensor_name("Where", self._tensor_name_set)
        node_name, self._node_name_mapping_dict = create_node_name(
            self._model.graph, "Where", self._node_name_mapping_dict
        )
        inp1 = (
            input_node_1.output[0]
            if isinstance(input_node_1, NodeProto)
            else input_node_1
        )
        inp2 = (
            input_node_2.output[0]
            if isinstance(input_node_2, NodeProto)
            else input_node_2
        )
        inp3 = (
            input_node_3.output[0]
            if isinstance(input_node_3, NodeProto)
            else input_node_3
        )
        where_node = make_node(
            "Where",
            inputs=[inp1, inp2, inp3],
            outputs=[output],
            name=node_name,
        )
        return where_node
