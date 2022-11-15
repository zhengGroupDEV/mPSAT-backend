"""
Description: convert CNNs to ONNX format
Author: Rainyl
License: Apache License 2.0
"""
import os
import random
import torch
import onnx
import onnxruntime
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict

from .model import CNN, CNN2D
from .config import ConfigCnn, ConfigCnn2D

def seed_everything(seed: int):
    print(f"set global seed to {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def replace_state_dict(ckpt: str, device: str):
    sd_new = OrderedDict()
    pl_state_dict = torch.load(ckpt, map_location=device)
    
    sd1 = pl_state_dict["state_dict"]
    for k in sd1:
        k1 = k.replace("model.", "")
        sd_new[k1] = sd1[k]
    return sd_new


def main(ckpt: str, save_to: str, model_name: str):
    device = "cpu"
    providers = ("CPUExecutionProvider", )
    seed_everything(42)
    if model_name == "cnn":
        # CNN
        conf = ConfigCnn("config/config_cnn.json")
        model = CNN(conf.in_channel, conf.num_class, dropout=conf.dropout)
        x = torch.randn(1, 1, 3600)
    elif model_name == "cnn2d":
        conf = ConfigCnn2D("config/config_cnn2d.json")
        model = CNN2D(
            in_channel=conf.in_channel,
            out_class=conf.num_class,
            dropout=conf.dropout,
        )
        x = torch.randn(1, 1, 480, 480)
    else:
        raise ValueError()
    state_dict = replace_state_dict(ckpt, device)
    model.load_state_dict(state_dict)
    model.eval()
    out = model(x)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        x,
        save_to,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    onnx_model = onnx.load(save_to)
    onnx.checker.check_model(onnx_model)  # type: ignore
    ort_session = onnxruntime.InferenceSession(save_to, providers=providers)

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", dest="src", type=str, required=True)
    parser.add_argument("-d", dest="dst", type=str, required=True)
    parser.add_argument("-m", "--model-name", dest="model_name", type=str, required=False)

    args = parser.parse_args()
    main(args.src, args.dst, args.model_name)
