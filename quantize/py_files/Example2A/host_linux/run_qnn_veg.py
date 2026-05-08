#!/usr/bin/env python
# coding: utf-8

# # QNN Model Prepare on Linux
# 
# The Qualcomm AI Engine Direct SDK allows clients to run ML models on HTP hardware. The following steps describe how to prepare the LLaVA models on Linux platforms for execution on Android.
# 
# Before continuing, ensure all steps from [README](readme.txt) are completed. 
# 
# This document uses the term Qualcomm Neural Network (QNN) and Qualcomm AI Engine Direct SDK interchangeably.

# ## Prerequisites
# 
# 1. Qualcomm AI Engine Direct SDK (with Ubuntu Linux support) version 2.31
# 2. Ubuntu 22.04 installation with required packages for QNN Tools
# 4. This notebook could be executed with Anaconda or a virtual environment (venv)
# 5. Qwen2-vl VEG `.onnx` files and their corresponding AIMET encodings (generated via AIMET workflow)
# 
# This work flow assumes that you have generated the Qwen2-VL model artifacts following the AIMET workflow (example1):
# 
# - VEG model and its AIMET encodings
# - `*.raw` file - input data for VEG model
# - `*.pkl` files per network - numpy object array saved as a Python pickle that contains data that is required as part of the model conversion step.

# ## Install the required python packages

# In[ ]:


pass  # pip install skipped


# ## Set up models and Qualcomm AI Engine Direct SDK variables

# In[ ]:




def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', required=True, help='Experiment name')
    args = parser.parse_args()

    import os
    # Self-contained ROOT detection: env BUNDLE_ROOT (set by wrapper) or
    # auto-derive from script location (../../../.. of this file's dir).
    BUNDLE_ROOT = os.environ.get('BUNDLE_ROOT') or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../..'))
    print(f'[run_qnn_veg] exp_name={args.exp_name}, BUNDLE_ROOT={BUNDLE_ROOT}')

    # Set up Qwen2-VL VEG models for on-target inference
    VEG_MODEL = f"{BUNDLE_ROOT}/results/{args.exp_name}/Example1A/veg_exports"

    # Set QNN_SDK_ROOT environment variable to the location of Qualcomm AI Engine Directory
    QNN_SDK_ROOT = '/opt/qcom/aistack/qairt/2.31.0.250130'
    # Create directory where artifacts will be exported
    EXPORT_DIR = f"{BUNDLE_ROOT}/results/{args.exp_name}/Example2A"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Check path to Qwen2-VL VEG_MODELS and QNN_SDK_ROOT
    assert os.path.exists(VEG_MODEL) == True, "VEG_MODEL path does not exist"
    assert os.path.exists(QNN_SDK_ROOT) == True, "QNN_SDK_ROOT path does not exist"
    os.environ['QNN_SDK_ROOT'] = QNN_SDK_ROOT


    # In[ ]:


    import sys
    sys.path.append('.')
    sys.path.append('..')
    from utilities.nsptargets import NspTargets

    # Set up nsp target specification
    # Android GEN3 and GEN4 are supported for this notebook
    nsp_target = NspTargets.Android.GEN4


    # ### Set up environment variables for the Qualcomm AI Direct SDK tools

    # In[ ]:


    import sys
    workfolder = os.getcwd()
    sys.path.append(os.path.join(os.path.dirname(workfolder), 'G2G'))
    sys.path.append(os.path.join(os.path.dirname(workfolder), 'G2G', 'split_onnx_utils'))
    sys.path.append(os.path.dirname(os.path.dirname(workfolder)))
    qnn_env = os.environ.copy()
    qnn_env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
    qnn_env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
    _qnn_bin = os.path.dirname(os.environ.get("PYTHON_QNN", "")) or os.path.dirname(sys.executable)
    qnn_env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + _qnn_bin + ":" + qnn_env["PATH"]
    qnn_env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
    qnn_env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
    os.environ = qnn_env


    # ## Convert the model from ONNX representation to QNN representation
    # 
    # The Qualcomm AI Engine Direct SDK `qnn-onnx-converter` tool converts a model from ONNX representation to its equivalent QNN representation in `A16W8` precision. The encoding files generated from the AIMET workflow are provided as an input to this step via the `--quantization_overrides model.encodings` option.
    # 
    # This step generates a `.cpp` file that represents the model as a series of QNN API calls and a `.bin` file that contains static data that is typically model weights and referenced by the `.cpp` file.
    # 
    # This step must be done independently for all models.

    # ### Generate model inputs list for VEG

    # In[ ]:


    veg_input_data = os.path.join(VEG_MODEL, "preprocessed_image.raw")
    cos_data = os.path.join(VEG_MODEL, "position_ids_cos.raw")
    sin_data = os.path.join(VEG_MODEL, "position_ids_sin.raw")
    mask_data = os.path.join(VEG_MODEL, "mask.raw")
    veg_input_file = os.path.join(VEG_MODEL, "veg_input_list.txt")

    with open(veg_input_file, "w") as f:
        # write to input_list.txt
        f.write("input:=" + veg_input_data + " position_ids_cos:=" + cos_data + " position_ids_sin:=" + sin_data + " mask:=" + mask_data)


    # # MHA2SHA

    # In[ ]:


    import subprocess

    mha2sha_root = os.path.join(os.path.dirname(workfolder), "G2G", "MHA2SHA")
    g2g_env = os.environ.copy()
    _qnn_bin = os.path.dirname(os.environ.get("PYTHON_QNN", "")) or os.path.dirname(sys.executable)
    g2g_env["PATH"] = _qnn_bin + ":" + g2g_env.get("PATH", "")
    g2g_env["PYTHONPATH"] = os.pathsep.join([g2g_env.get("PYTHONPATH", ""), os.path.join(mha2sha_root, "src/python")])
    g2g_env["PATH"] = os.pathsep.join([g2g_env.get("PATH", ""), os.path.join(mha2sha_root, "bin")])
    print(f"MHA2SHA tool root set to: {mha2sha_root}")

    print(g2g_env["PYTHONPATH"])
    sha_name = "veg_sha"
    sha_folder = "exports/sha_output"
    def thread_g2g():
        os.makedirs(sha_folder, exist_ok=True)
        proc = subprocess.Popen([
            "mha2sha-onnx-converter",
            "--model-name", sha_name,
            "--exported-model-encoding-path", VEG_MODEL + "/veg.encodings",
            "--exported-model-path", VEG_MODEL + "/veg.onnx",
            "--nchw-aligned",
            "--mha-conv",
            "--prepared-model",
            "--handle-rope-ops",
            "--create-input-lists",
            "--log-level", "verbose",
            "--sha-export-path", sha_folder],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=g2g_env)
        output, error = proc.communicate()
        print(output.decode(), error.decode())

    thread_g2g()
    print(f"All mha2sha convert done.")


    # ### Convert VEG
    # Expected execution time: 3min ~ 4min

    # In[ ]:


    import subprocess

    os.makedirs(os.path.join(EXPORT_DIR, "converted_veg"), exist_ok=True)
    proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-onnx-converter",
                      "-o", EXPORT_DIR + "/converted_veg/veg.cpp",
                       "--input_network", os.path.join(sha_folder, sha_name + ".onnx"),
                       "--input_list", VEG_MODEL + "/veg_input_list.txt",
                       "--act_bitwidth", "16",
                       "--bias_bitwidth", "32",
                       "--quantization_overrides", os.path.join(sha_folder, sha_name + ".encodings")
                     ],stdout=subprocess.PIPE, stderr=subprocess.PIPE,env=qnn_env)
    output, error = proc.communicate()
    print(output.decode(),error.decode())


    # ## QNN model library
    # 
    # The  Qualcomm AI Engine Direct SDK `qnn-model-lib-generator` compiles the model `.cpp` and `.bin` files into a shared object library for a specific target. This example generates a shared object library for x86_64-linux target.
    # 
    # The inputs to this stage are the `model.cpp` and `model.bin` files generated in the previous step.
    # 
    # ### Generate the VEG model library
    # Expected execution time: ~5 minutes

    # In[ ]:


    proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-model-lib-generator",
                            "-c", EXPORT_DIR + "/converted_veg/veg.cpp",
                            "-b", EXPORT_DIR + "/converted_veg/veg.bin",
                            "-t", "x86_64-linux-clang",
                            "-o", EXPORT_DIR + "/converted_veg"
                            ],stdout=subprocess.PIPE, stderr=subprocess.PIPE,env=qnn_env)
    output, error = proc.communicate()
    print(output.decode(),error.decode())


    # ## QNN HTP weight sharing context binary
    # 
    # The  Qualcomm AI Engine Direct SDK `qnn-context-binary-generator` tool creates a QNN context binary applicable to the QNN HTP backend. This binary can be deployed to run on a Snapdragon 8 Gen 4 device that runs Android. This step requires the model shared object library from the previous step and the `libQnnHtp.so` library, available in the Qualcomm AI Engine Direct SDK.
    # 
    # Provide additional options that pertain to the QNN HTP backend by passing the `libQnnHtpBackendExtensions.so` library that implements extensions for the QNN HTP backend. The library is available in the Qualcomm AI Engine Direct SDK. The library and configurations are provided as a `.json` format as shown below. Documentation on backend extensions and configuraton parameters is available in the Qualcomm AI Engine Direct SDK Documents.

    # In[ ]:


    import json

    # Create tmp directory for config files
    CONFIG_DIR = os.path.join(EXPORT_DIR, "configs")
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # HTP backend extensions config file (htp_backend_extensions.json) example
    htp_backend_extensions_data = {"backend_extensions": {"shared_library_path": "libQnnHtpNetRunExtensions.so", "config_file_path": os.path.join(CONFIG_DIR, "htp_config.json")}}

    # HTP backend config file (htp_config.json) example
    htp_backend_config_data = {
        "graphs": [{
            "vtcm_mb": 8,
            "graph_names": [],
            "O": 3.0,
            "fp16_relaxed_precision": 0
        }],
        "devices": [{
            "soc_id": nsp_target.soc_id,
            "dsp_arch": nsp_target.dsp_arch,
            "cores": [{
                "core_id": 0,
                "perf_profile": "burst",
                "rpc_control_latency": 100
            }]
        }]
    }


    # In[ ]:


    # Create a path under the models directory for serialized binaries
    os.makedirs(os.path.join(EXPORT_DIR, "serialized_binaries"), exist_ok=True)


    # ### Generate the QNN context binary for VEG 
    # Expected execution time: ~ 2.5 Hours

    # In[ ]:


    # write the config files to a temporary location
    htp_backend_config_data["graphs"][0]["graph_names"] = ["veg"]
    with open(os.path.join(CONFIG_DIR, "htp_backend_extensions.json"),'w') as f:
        f.write(json.dumps(htp_backend_extensions_data, indent=4))
    with open(os.path.join(CONFIG_DIR, "htp_config.json"),'w') as f:
        f.write(json.dumps(htp_backend_config_data, indent=4))

    proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qnn-context-binary-generator",
                                 "--model", EXPORT_DIR + "/converted_veg/x86_64-linux-clang/libveg.so",
                                 "--backend", "libQnnHtp.so",
                                 "--output_dir",  EXPORT_DIR + "/serialized_binaries",
                                 "--binary_file", "veg.serialized",
                                 "--config_file", os.path.join(CONFIG_DIR, "htp_backend_extensions.json")
                            ],stdout=subprocess.PIPE, stderr=subprocess.PIPE,env=qnn_env)
    output, error = proc.communicate()
    print(output.decode(),error.decode())


    #  
    # Copyright (c) 2024 Qualcomm Technologies, Inc. and/or its subsidiaries.



if __name__ == "__main__":
    main()
