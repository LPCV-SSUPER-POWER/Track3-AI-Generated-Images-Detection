from qwen_vl_utils import process_vision_info
from glob import glob
import random
import contextlib
import os
import torch
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.layer_output_utils import LayerOutputUtil, NamingScheme

def dump_quantsim(dummy_input, quantsim, output_dir):
    # Use quantsim model to get quantsim layer-outputs
    quantsim_layer_output_util = LayerOutputUtil(model=quantsim.model,
                                                dir_path=output_dir,
                                                naming_scheme=NamingScheme.ONNX,
                                                dummy_input=dummy_input,
                                                onnx_export_args={})
    quantsim_layer_output_util.generate_layer_outputs(dummy_input)


def load_quantsim(config, dummy_input, pth, encodings_path):
    from veg_utils.aimet_quantsim import matmul_exceptions, enable_concat_input_quantizers,buffer_concat_exceptions
    model = torch.load(pth)
    # Use same arguments as that were used for the exported QuantSim model. For sake of simplicity only mandatory arguments are passed below.
    quantsim = QuantizationSimModel(
        model=model,
        quant_scheme=config.quant_scheme,
        dummy_input=dummy_input,
        default_output_bw=config.activation_bit_width,
        default_param_bw=config.parameter_bit_width,
        in_place=config.in_place,
        config_file=config.config_file,
    )
    enable_concat_input_quantizers(quantsim)
    if config.use_16_8_matmul:
        matmul_exceptions(quantsim)
    # Load exported encodings into quantsim object
    load_encodings_to_sim(quantsim, encodings_path)

    # Post-calibration exceptions
    buffer_concat_exceptions(quantsim) # overwrite encodings to guarantee match
    return quantsim


def data_preprocess(processor, img_path, inp_h=640, inp_w=640, prompt=''):

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": img_path,
                "resized_height": inp_h,
                "resized_width": inp_w,
            },
            {
                "type": "text",
                "text": prompt
            },
        ],
    }]
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    # print(image_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def get_dummy_input(processor, inp_h=640, inp_w=640, img_path="./images/demo512x342.jpeg"):
    inputs = data_preprocess(processor, img_path, inp_h, inp_w)
    return inputs["pixel_values"], inputs["image_grid_thw"]


def re_export_onnx_graph_and_weights(src, dst_dir, name):
    import onnx
    os.makedirs(dst_dir, exist_ok=True)
    out_onnx = os.path.join(dst_dir, name + '.onnx')
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(src)), out_onnx, location=name + ".bin", size_threshold=0, save_as_external_data=True)


def load_and_preprocess_images(image_root, num_samples, processor, inp_h=640, inp_w=640):
    image_list = glob(image_root + '/*.jpg')
    image_list = random.sample(image_list, num_samples)
    preprocessed_images = []
    for image_file in image_list:
        '''
        1. open the image
        2. expand to square which hf doesn't do
        3. pass it through the image processor from qwen2-vl, this basically converts the RGB into a tensor and gives (1,3,342,512) image tensor which is
        then passed through the VEG to get the embeddings projected in the language space.
        '''
        inputs = data_preprocess(processor, image_file, inp_h, inp_w)
        image_tensor = inputs["pixel_values"]
        preprocessed_images.append(image_tensor)
    return preprocessed_images


def to_device(device, *ts):
    out = []
    for t in ts:
        out.append(t.to(device))
    return out
