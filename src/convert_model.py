"""
Description: convert CNN to ONNX format
Author: Rainyl
License: Apache License 2.0
"""
import torch
import onnx
import onnxruntime
import numpy as np
from argparse import ArgumentParser

from src.model import CNN


def main(ckpt: str, save_to: str):
    device = "cpu"
    x = torch.randn(1, 1, 480, 480)
    model = CNN(1, 120, dropout=0.2)
    model.load_state_dict(torch.load(ckpt, map_location=device))
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
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(save_to)

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

    args = parser.parse_args()
    main(args.src, args.dst)
