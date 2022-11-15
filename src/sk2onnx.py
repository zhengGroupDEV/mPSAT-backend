"""
Description: convert sklearn models to ONNX format
Author: Rainyl
License: Apache License 2.0
"""
from argparse import ArgumentParser, Namespace
import inspect
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import numpy as np
import pickle
from pathlib import Path


def main(args: Namespace):
    with open(args.src, "rb") as f:
        clf = pickle.load(f)
    X = np.random.normal(size=(1, 3600))
    func_names = [f[0] for f in inspect.getmembers(clf, inspect.isroutine)]
    if "predict_proba" in func_names:
        y_score = clf.predict_proba(X)  # type: ignore
    else:
        y_score = clf.decision_function(X)  # type: ignore
    initial_type = [("input", FloatTensorType([None, 3600]))]
    save_to = Path(args.dst)
    options = {id(clf): {"zipmap": False}}
    onx = to_onnx(
        clf,
        X,
        options=options,
        initial_types=initial_type,
        target_opset=15,
    )
    with open(save_to, "wb") as f:
        f.write(onx.SerializeToString())  # type: ignore

    sess = ort.InferenceSession(
        str(save_to.absolute()), providers=("CPUExecutionProvider",)
    )
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: X.astype(np.float32)})[1]

    np.testing.assert_allclose(y_score, pred_onx, rtol=1e-3, atol=1e-5)

    print(f"Convert and check finished, all correct!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", dest="src", type=str, required=True)
    parser.add_argument("-d", dest="dst", type=str, required=True)

    args = parser.parse_args()
    main(args)
