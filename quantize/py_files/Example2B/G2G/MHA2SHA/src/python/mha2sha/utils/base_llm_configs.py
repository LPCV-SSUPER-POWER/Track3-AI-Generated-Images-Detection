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

_llama2_flags = {
    "create_input_lists": False,
    "disable_auto_attn_finder": False,
    "gqa_model": False,
    "handle_rope_ops": True,
    "handle_past_key_value": True,
    "llm_model": True,
    "mha_conv": False,
    "nchw_aligned": False,
    "no_verification": False,
    "prepared_model": True,
    "replace_linear_with_conv": True,
    "strict": True,
}

_all_off_flags = {arg_name: False for arg_name in _llama2_flags.keys()}
_lora_flags = {
    "lora_model": True,
    "nchw_aligned": True,
    "mha_conv": True,
    "lora_alpha_from_input": True,
}

base_llms_to_flags = {
    "llama2": _llama2_flags,
    "llama2_lora": _llama2_flags | _lora_flags,
    "llama3": _llama2_flags | {"gqa_model": True},
    "sd_2.1": _all_off_flags | {"nchw_aligned": True, "mha_conv": True},
    "sd_2.1_lora": _all_off_flags | _lora_flags,
}
