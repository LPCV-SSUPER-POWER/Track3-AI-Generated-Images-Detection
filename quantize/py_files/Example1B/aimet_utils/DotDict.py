# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. 
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

class DotDict(dict):
    """A dictionary that supports dot notation as well as dictionary access notation."""
    def __getattr__(self, attr):
        return self.get(attr)
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        del self[key]
    
    @staticmethod
    def from_dict(data):
        """Recursively converts a dictionary to a DotDict."""
        if not isinstance(data, dict):
            return data
        return DotDict({k: DotDict.from_dict(v) for k, v in data.items()})

def custom_nb_config(config):
    context = {
        'context_length': config['model']['context_length']
    }

    return interpolate_config(config, context)

# Function to recursively replace placeholders in the config
def interpolate_config(config, context):

    if isinstance(config, dict):
        return {k: interpolate_config(v, context) for k, v in config.items()}
    elif isinstance(config, list):
        return [interpolate_config(item, context) for item in config]
    elif isinstance(config, str):
        return config.format(**context)
    else:
        return config
