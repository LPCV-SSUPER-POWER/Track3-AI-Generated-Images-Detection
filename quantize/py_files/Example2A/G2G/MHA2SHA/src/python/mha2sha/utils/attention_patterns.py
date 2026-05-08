# -----------------------------------------------------------------------------
#
# Qualcomm Technologies, Inc. Proprietary
# (c) 2022-24 Qualcomm Technologies, Inc. All rights reserved.
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

attention_patterns = [
    {
        "pattern": ["MatMul", "Div", "Add", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Div", "Where", "Add", "Softmax", "MatMul"],
    },
    {
        "pattern": ["MatMul", "Div", "Add", "Add", "Softmax", "MatMul"],
    },
    {
        "pattern": ["MatMul", "Add", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Div", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Where", "Softmax", "MatMul"]
    },
    {
        "pattern": ["MatMul", "Mul", "Softmax", "MatMul"]  # SD 2.1
    },
    {
        "pattern": ["MatMul", "Div", "Add", "Softmax", "Cast", "Cast", "MatMul"]  # LLaMA v2 SS
    },
    {
        "pattern": ["MatMul", "Div", "Transpose", "Add", "Transpose", "Softmax", "MatMul"]  # LLaMA v2 BERT
    }
]
