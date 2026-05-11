#!/usr/bin/env python
# coding: utf-8

# # QNN Model Prepare on Linux
# 
# The Qualcomm AI Engine Direct SDK allows clients to run ML models on HTP hardware. The following steps describe how to prepare the Qwen2 models on Linux platforms for execution on Android.
# 
# Before continuing, ensure all steps from [README](README.md) are completed. 
# 
# This document uses the term Qualcomm Neural Network (QNN) and Qualcomm AI Engine Direct SDK interchangeably.
# 
# 
# # Prerequisites
# 
# 1. Qualcomm AI Engine Direct SDK (with Ubuntu Linux support)
# 2. Ubuntu 22.04 installation with required packages for QNN Tools
# 3. Android Platform tools version 31 or greater
# 4. This notebook could be executed with Anaconda (with the supplied environment.yaml) or a virtual environment(venv)
# 5. Qwen2-VL language `.onnx` files and their corresponding AIMET encodings (generated via AIMET workflow)
# 
# This work flow assumes that you have generated the Qwen2-VL model artifacts following the AIMET Qwen2-VL workflow (example1):
# 
# - Qwen2-VL language model and its AIMET encodings
# - `*.pkl` files per network - numpy object array saved as a Python pickle that contains data that is required as part of the model conversion step.
# 
# ![dir_struct](../jupyter_notebook_assets/nb1_output_dir_contents.png "Overall directory Structure from notebook 1") ![onnx_dir_struct](../jupyter_notebook_assets/onnx_dir_struct.png "Snapshot of file contents of onnx folder from notebook 1")
# 

# ## Set up the Qualcomm AI Engine Direct SDK
# 
# The following steps configure the Qualcomm AI Engine Direct SDK, which enables running Qwen2 on the device. 
# Execute the following on an Ubuntu 22.04 terminal. 
# 
# **NOTE:** These steps require sudo or root privileges.
# 
# 1. After setting up Python and pip in Ubuntu, check QNN tool dependencies. 
# 2. Set the `QNN_SDK_ROOT` environment variable to the location of the Qualcomm AI Engine Directory. For **Linux**, `export QNN_SDK_ROOT="./assets/qnn"`
# 3. Check and install Linux dependencies.
# 
#     ```
#     source $QNN_SDK_ROOT/bin/check-linux-dependency.sh
#     sudo apt-get install -y libtinfo5
#     ```

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
    print(f'[run_qnn_llm] exp_name={args.exp_name}, BUNDLE_ROOT={BUNDLE_ROOT}')
    # Set QNN_SDK_ROOT environment variable to the location of Qualcomm AI Engine Directory
    QNN_SDK_ROOT = '/opt/qcom/aistack/qairt/2.31.0.250130' 
    # Check path QNN_SDK_ROOT
    assert os.path.exists(QNN_SDK_ROOT) == True,"QNN_SDK_ROOT path does not exist"
    os.environ['QNN_SDK_ROOT'] = QNN_SDK_ROOT


    # ### Install the required python packages

    # In[ ]:


    pass  # pip install skipped


    # ## Set up models and Qualcomm AI Engine Direct SDK variables

    # In[ ]:


    import subprocess
    import concurrent.futures
    import time
    from pathlib import Path
    # setup whether using multithread or single thread to compile
    go_parallel = False

    workfolder = os.getcwd()
    # Set up environment variable to reference Qwen2_MODELS
    QWEN2_MODELS = f"{BUNDLE_ROOT}/results/{args.exp_name}/Example1B/export"
    # Check path QWEN2_MODELS
    assert os.path.exists(QWEN2_MODELS) == True,"QWEN2_MODELS path does not exist"


    # # Workflow for Qwen2 models(language part of Qwen2-VL)
    # 
    # 
    # All the models and encodings are processed independently via different executable QNN utilities available in the Qualcomm AI Engine Direct SDK.
    # 
    # To prepare Qwen2 models(language part of Qwen2-VL) for inference, the QNN executable utilities require an Ubuntu 22.04 environment
    # 
    # 1. Split the onnx model into several small onnx models.
    # 2. Apply MHA2SHA transformation to convert all attention block MHAs to SHAs.
    # 3. Convert the `.onnx` files to their equivalent QNN representation.
    # 4. Generate the QNN model quantized libraries.
    # 5. Generate the QNN context binaries for the QNN HTP backend.
    # 
    # After preparing the Qwen2 models(language part of Qwen2-VL) for inference, the next step is to execute the QNN context binaries for inference on a Snapdragon Android
    # 
    # 
    # ![QNN Work flow](../jupyter_notebook_assets/qnn-workflow.png)

    # In[ ]:


    import sys
    sys.path.append('.')
    sys.path.append('..')
    sys.path.append(os.path.join(os.path.dirname(workfolder), 'G2G'))
    sys.path.append(os.path.join(os.path.dirname(workfolder), 'G2G', 'split_onnx_utils'))
    sys.path.append(os.path.dirname(os.path.dirname(workfolder)))
    from utilities.nsptargets import NspTargets
    from utilities.profiler import event_marker

    # Set up nsp target specification
    # Android GEN2 or Gen4 or higher is supported for this notebook
    nsp_target = NspTargets.Android.GEN4

    CL = 2048
    ARNs = [1, 128]
    EXPORT_AR = 1073
    EXPORT_CONTEXT_LENGTH = 2048
    # onnx_name must match the filename produced by Example1B
    model_name = args.exp_name.lower().replace(".", "p").replace("-", "_") + "_merged_stage2"
    onnx_name = f"{model_name}_AR{EXPORT_AR}"
    num_splits = 1

    splits = range(1, num_splits+1)
    arn_list = [ arn for arn in ARNs for i in splits ]
    split_idxs = [i for arn in ARNs for i in splits]
    print('All task list:', [f"ar{arn}-{n}" for arn,n in zip(arn_list,split_idxs)])


    # # Prepare Qwen2 models(language part of Qwen2-VL) for Inference
    # 
    # The following section uses the Qualcomm AI Engine Direct SDK to prepare Qwen2 models(language part of Qwen2-VL) for on-target inference.

    # In[ ]:


    os.makedirs(f"{workfolder}/assets/models_ar_n", exist_ok=True)

    import change_hardcoding
    def gen_ar(arn):
        try:
            change_hardcoding.execute(
                    f"{QWEN2_MODELS}",
                    f"{workfolder}/assets/models_ar_n/ar{arn}-cl{CL}",
                    [f" {EXPORT_AR},{arn}",f" -{EXPORT_AR},-1",f" {EXPORT_CONTEXT_LENGTH},{CL}",f" {EXPORT_CONTEXT_LENGTH-EXPORT_AR},{CL-arn}"]
                    )
        except Exception as e:
            print(e)
            raise

    with event_marker(f'prepare-export'):
            for _arg in ARNs:
                gen_ar(_arg)

    print(f"Prepare AR128 AR1 export done.")


    # ## Preprocess ONNX 
    # 
    # Prior to utilizing the QNN tool chain to compile and generate the context binary for Qwen2 we need to split the model and generate the following artifacts
    # - ONNX file for each split of the model
    # - input vectors for each split
    # - golden output vectors for each split
    # 
    # We need to specify the following parameters to proceed with execution of the notebook and generate all necessary artifacts
    # - number of splits of the model
    # - path to Qwen2 onnx file
    # - path to Qwen2 encodings file
    # - path to *.pkl files 
    #   
    # 
    # ![Split](../jupyter_notebook_assets/ModelSplit.png)

    # ### Set up environment variables for the Qualcomm AI Direct SDK tools

    # In[ ]:


    import os
    import utils

    qnn_env = os.environ.copy()
    qnn_env["QNN_SDK_ROOT"] = QNN_SDK_ROOT
    qnn_env["PYTHONPATH"] = QNN_SDK_ROOT + "/benchmarks/QNN/:" + QNN_SDK_ROOT + "/lib/python"
    _qnn_bin = os.path.dirname(os.environ.get("PYTHON_QNN", "")) or os.path.dirname(sys.executable)
    qnn_env["PATH"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + _qnn_bin + ":" + qnn_env["PATH"]
    qnn_env["LD_LIBRARY_PATH"] = QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
    qnn_env["HEXAGON_TOOLS_DIR"] = QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
    qnn_env["NUM_LAYERS_PER_SPLIT"] = "28"
    qnn_env["LLM"] = "1"
    qnn_env["split_embedding"] = "0"
    qnn_env["split_lmhead"] = "0"
    os.environ = qnn_env


    # ### Split Onnx export
    # 
    # This step splits a model into multiple parts based on the number of splits specified.
    # 
    # Expected execution time: ~< 10 minutes

    # In[ ]:


    def thread_split(arn):
        try:
            name = f"ar{arn}-cl{CL}"
            model_export = f"{workfolder}/assets/models_ar_n"
            model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
            os.makedirs(model_artifact, exist_ok = True)

            # create symlink to export
            symlink_src = os.path.join(model_artifact, 'src')
            symlink_path = Path(symlink_src)
            if symlink_path.is_symlink():
                os.unlink(symlink_src)
            os.symlink(src = os.path.join(model_export, name), dst = symlink_src)

            os.makedirs(f"{model_artifact}/split_onnx", exist_ok = True)
            TEST_VECTOR_PICKLE_TYPE = "pkl"
            print(f"Starting {onnx_name}.onnx")
            utils.split_onnx(onnxfile = f"{model_artifact}/src/onnx/{onnx_name}.onnx", modelname = name,
                            pickle_filedir = os.path.join(model_export, f"ar{arn}-cl{CL}/test_vectors"),
                            num_splits = num_splits, output_dir = model_artifact, split_embedding = False,
                            encoding_file = f"{model_artifact}/src/onnx/{onnx_name}.encodings",using_qairt_workflow = True
                            )
            print(f"Ending {onnx_name}.onnx")
        except Exception as e:
            print(e)
            raise

    with event_marker(f'split-onnx'):
            for _arg in ARNs:
                thread_split(_arg)

    print(f"All onnx model splitted.")


    # ### Convert attention layers from MHA to SHA
    # 
    # The `mha2sha-onnx-converter` tool converts a model from MHA representation to its equivalent SHA representation. The encoding files generated from the AIMET workflow are provided as an input to this step via the `--exported-model-encoding-path` option.
    # 
    # This step generates a new `.onnx` file that represents the model in SHA format.
    # 
    # Expected execution time: ~20 minutes

    # In[ ]:


    mha2sha_root = os.path.join(os.path.dirname(workfolder), "G2G", "MHA2SHA")
    g2g_env = os.environ.copy()
    _qnn_bin = os.path.dirname(os.environ.get("PYTHON_QNN", "")) or os.path.dirname(sys.executable)
    g2g_env["PATH"] = _qnn_bin + ":" + g2g_env.get("PATH", "")
    g2g_env["PYTHONPATH"] = os.pathsep.join([g2g_env.get("PYTHONPATH", ""), os.path.join(mha2sha_root, "src/python")])
    g2g_env["PATH"] = os.pathsep.join([g2g_env.get("PATH", ""), os.path.join(mha2sha_root, "bin")])
    print(f"MHA2SHA tool root set to: {mha2sha_root}")

    def thread_g2g(arn,split):
        try:
            model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
            split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
            name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
            os.makedirs(split_work_dir, exist_ok = True)
            sha_folder = f"{split_work_dir}/sha_output/"
            os.makedirs(sha_folder, exist_ok = True)
            name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
            print(f"mha2sha-onnx-converter {name} running...")
            args=["mha2sha-onnx-converter",
                                "--sha-export-path", sha_folder,
                                "--model-name", name,
                                "--exported-model-encoding-path", f"{model_artifact}/src/onnx/{onnx_name}.encodings",
                                "--exported-model-path", f"{model_artifact}/split_onnx/{name}.onnx",
                                # "--base-llm", "llama3",
                                "--llm-model",
                                "--handle-rope-ops",
                                "--handle-past-key-value",
                                "--mha-conv",
                                "--gqa-model",
                                "--nchw-aligned"]
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=g2g_env)
            output, error = proc.communicate()
            print(output.decode(),error.decode())
            print(f"mha2sha-onnx-converter {name} done.")
        except Exception as e:
            print(e)
            raise
    with event_marker(f'mha2sha'):
            for _a1, _a2 in zip(arn_list, split_idxs):
                thread_g2g(_a1, _a2)

    print(f"All mha2sha convert done.")


    # ## Convert the model from ONNX representation to QNN DLC representation
    # 
    # The Qualcomm AI Engine Direct SDK `qairt-converter` tool converts a model from ONNX representation to its equivalent QNN DLC representation. The encoding files generated from the AIMET workflow are provided as an input to this step via the `–quantization_overrides model.encodings` option.
    # 
    # This step generates a `.dlc` file that represents the model as a series of QNN API calls.
    # 
    # Expected execution time: ~< 20 minutes

    # In[ ]:


    def thread_convert(arn,split):
        try:
            model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
            split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
            name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
            os.makedirs(split_work_dir, exist_ok = True)
            out_dir = os.path.join(split_work_dir, "converted_model")
            os.makedirs(out_dir, exist_ok = True)

            # create symlink to export
            for src in [f"input_list_{name}.txt",f"test_inputs_{name}"]:
                symlink_input = os.path.join(split_work_dir, src)
                symlink_path = Path(symlink_input)
                if symlink_path.is_symlink():
                    os.unlink(symlink_input)
                os.symlink(src = os.path.join(model_artifact, src), dst = symlink_input)
            input_onnx=f"{split_work_dir}/sha_output/{name}.onnx"
            quantization_overrides= f"{split_work_dir}/sha_output/{name}.encodings"

            args = [QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-converter",
                            "--input_network", input_onnx,
                            "--quantization_overrides", quantization_overrides,
                            "-o", f'{out_dir}/{name}.dlc'
                            ]
            options = utils.get_input_layout(input_onnx, using_qairt_workflow = True)
            for entry in options:
                args+=entry

            proc = subprocess.Popen(args, stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = qnn_env)
            output, error = proc.communicate()
            print(output.decode(), error.decode())
            print(f"qairt-converter {name} done!")
        except Exception as e:
            print(e)
            raise

    with event_marker(f'convert-onnx'):
            for _a1, _a2 in zip(arn_list, split_idxs):
                thread_convert(_a1, _a2)

    print(f"All qairt-converter done.")


    # ##  Quantized QNN DLC model
    # 
    # The  Qualcomm AI Engine Direct SDK `qairt-quantizer` compiles the model `.dlc` and input`.raw` files into a `model.quantized.dlc` file.
    # 
    # The inputs to this stage are the input raw files &  `model.dlc` generated in the previous step.
    # 
    # Expected execution time: ~< 10 minutes

    # In[ ]:


    def thread_genlib(arn,split):
        try:
            model_artifact = f"{workfolder}/assets/artifacts/ar{arn}-cl{CL}/"
            split_work_dir = os.path.join(model_artifact,f"{split}_of_{num_splits}")
            name = f"ar{arn}-cl{CL}_{split}_of_{num_splits}"
            os.chdir(split_work_dir)
            out_dir = os.path.join(split_work_dir,"compiled_model")
            os.makedirs( os.path.join(split_work_dir,"compiled_model"), exist_ok = True)

            float_dlc_file = os.path.join(split_work_dir, "converted_model", f'{name}.dlc')
            quantized_dlc_file = os.path.join(out_dir, f'{name}_quantized.dlc')
            ip_list_file = os.path.join(model_artifact, f'input_list_{name}.txt')

            proc = subprocess.Popen([QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-quantizer",
                                    "--input_dlc", float_dlc_file,
                                    "--input_list", ip_list_file,
                                    "--output_dlc", quantized_dlc_file,
                                    "--act_bitwidth", "16",
                                    "--bias_bitwidth", "32"
                                    ],stdout = subprocess.PIPE, stderr = subprocess.PIPE, env = qnn_env)
            output, error = proc.communicate()
            print(output.decode(), error.decode())
            print(f"qairt-quantizer {name} done!")
            os.chdir(workfolder)
        except Exception as e:
            print(e)
            raise

    with event_marker(f'qairt-quantizer'):
            for _a1, _a2 in zip(arn_list, split_idxs):
                thread_genlib(_a1, _a2)

    print(f"All qairt-quantizer done.")


    # ## QNN HTP weight sharing context binary
    # 
    # The  Qualcomm AI Engine Direct SDK `qnn-context-binary-generator` tool creates a QNN context binary applicable to the QNN HTP backend. This binary can be deployed to run on a Snapdragon 8 Gen4 device that runs Android. This step requires the ar128 and ar1 quantized DLCs from the previous step and the `libQnnHtp.so` library, available in the Qualcomm AI Engine Direct SDK.
    # 
    # Provide additional options that pertain to the QNN HTP backend by passing the `libQnnHtpBackendExtensions.so` library that implements extensions for the QNN HTP backend. The library is available in the Qualcomm AI Engine Direct SDK.
    # 
    # ### Define Htp Perf Setting

    # In[ ]:


    import os
    import json

    def make_config_file(index, folder, src_graphs, soc_id=43, dsp_arch="v73"):
        htp_config_json = os.path.join(folder, f"HtpConfigFile_API_{index}.json")
        perf_config_json = os.path.join(folder, f"PerfSetting_API_{index}.conf")

        soc_id = int(soc_id)
        with open(htp_config_json, 'w') as f:
            config = {
                "backend_extensions": {
                    "shared_library_path": "libQnnHtpNetRunExtensions.so",
                    "config_file_path": f"{perf_config_json}"
                }
            }

            json.dump(config, f, indent=4)

        with open(perf_config_json,'w') as f:
            config = {
                "graphs": [{
                    "O": 3.0,
                    "vtcm_mb": 8,
                    "graph_names": src_graphs,
                    "fp16_relaxed_precision": 0
                }],
                "devices": [
                    {
                        "soc_id": soc_id,
                        "dsp_arch": dsp_arch,
                        "cores": [
                            {
                                "perf_profile": "burst",
                                "rpc_control_latency": 100
                            }
                        ],
                        "pd_session": "unsigned"
                    }
                ],
                "context": {
                        "weight_sharing_enabled": len(src_graphs) > 1
                },
                "memory": {
                        "mem_type": "shared_buffer"
                }
            }
            json.dump(config, f, indent = 4)


    # ### Compile context binary
    # Expected execution time: ~20 minutes

    # In[ ]:


    import subprocess

    soc_id = nsp_target.soc_id
    dsp_arch = nsp_target.dsp_arch

    def thread_gen_ws_cb(i):
        try:
            ar128_src = f"{workfolder}/assets/artifacts/ar128-cl{CL}/"
            ar1_src = f"{workfolder}/assets/artifacts/ar1-cl{CL}/"
            output_dir = f"{workfolder}/assets/artifacts/ar128-ar1-cl{CL}_conf_files/"
            ctx_output_dir = f"{workfolder}/assets/artifacts/ar128-ar1-cl{CL}/"
            os.makedirs(output_dir, exist_ok = True)
            os.makedirs(ctx_output_dir, exist_ok = True)

            src1_split_folder = os.path.join(ar128_src, f"{i}_of_{num_splits}", "compiled_model")
            src2_split_folder = os.path.join(ar1_src, f"{i}_of_{num_splits}", "compiled_model")

            src1_graph_name = f"ar128-cl{CL}_{i}_of_{num_splits}"
            src1_q_dlc = os.path.join(src1_split_folder, f"{src1_graph_name}_quantized.dlc")
            src2_graph_name = f"ar1-cl{CL}_{i}_of_{num_splits}"
            src2_q_dlc = os.path.join(src2_split_folder, f"{src2_graph_name}_quantized.dlc")

            graph_list = [src1_graph_name, src2_graph_name]
            make_config_file(i, output_dir, graph_list, soc_id, dsp_arch)

            cmd = ["qnn-context-binary-generator",
                    "--log_level=verbose",
                    "--backend","libQnnHtp.so",
                    "--model", "libQnnModelDlc.so",
                    "--input_output_tensor_mem_type", "memhandle",
                    "--output_dir", ctx_output_dir,
                    "--config_file",f"{output_dir}/HtpConfigFile_API_{i}.json",
                    "--binary_file", f"weight_sharing_model_{i}_of_{num_splits}.serialized",
                    "--dlc_path", f"{src1_q_dlc},{src2_q_dlc}"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=qnn_env)
            output, error = proc.communicate()
            print(output.decode(), error.decode())
            print(f'#{i} weight sharing model generated')
        except Exception as e:
            print(e)
            raise

    with event_marker(f'gen-binary'):
            for _arg in splits:
                thread_gen_ws_cb(_arg)

    print(f"All weight shared qnn-context-binary generated.")


    # ### Save profiling stats

    # In[ ]:


    from utilities.profiler import EventProfiler
    EventProfiler().report()
    EventProfiler().json_dump(os.path.join(workfolder, 'assets/profiling_stats.json'))


    # In[ ]:







if __name__ == "__main__":
    main()

    # Copy final artifacts to the results/ directory
    import os, shutil, glob, sys
    # Self-contained ROOT detection (env BUNDLE_ROOT or auto from script location)
    BUNDLE_ROOT = os.environ.get('BUNDLE_ROOT') or os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../..'))
    exp_name = sys.argv[sys.argv.index("--exp-name") + 1] if "--exp-name" in sys.argv else None
    if exp_name:
        src_pattern = f"assets/artifacts/ar128-ar1-cl*/weight_sharing_model_*.serialized.bin"
        dst_dir = f"{BUNDLE_ROOT}/results/{exp_name}/Example2B"
        os.makedirs(dst_dir, exist_ok=True)
        for src in glob.glob(src_pattern):
            dst = os.path.join(dst_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"[run_qnn_llm] Copied {src} -> {dst}")
