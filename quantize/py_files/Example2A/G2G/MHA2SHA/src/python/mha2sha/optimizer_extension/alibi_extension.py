
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
from typing import Any
from onnx.onnx_pb import NodeProto

mha2sha_hf_model_optimizer = Any  # Causes circular import

class AlibiExtension:
    """Extension helpers for mha2sha_optimizer for ALiBi position ids"""

    def __init__(self, mha2sha_optim: mha2sha_hf_model_optimizer) -> None:
        """Initalizes an instance based on the MHA2SHA optimizer provided.

        Args:
            mha2sha_optim:
                MHA2SHAOptimizer instance holding the model loader and model info. 
        """
        self.mha2sha_optim = mha2sha_optim

    def get_curr_head_slice_inp(self, original_inp: str, head_num: int) -> NodeProto:
        """Returns the Slice operation for a ALiBi Add.

        Args:
            original_inp:
                Original input into the Add that is not from the attention mask Div.
            head_num:
                Which head num we are wanting to slice.

        Returns:
            Created Slice Op.
        """

        slice_node, slice_init = self.mha2sha_optim._op_factory.get_slice_op(
            original_inp, start=head_num, end=head_num + 1, axis=1
        )
        self.mha2sha_optim.model.graph.node.append(slice_node)
        self.mha2sha_optim.model.graph.initializer.extend(slice_init)
        return slice_node
