import torch
import time
import argparse  # Step 1: Import argparse
from utilities import Engine

# Step 2: Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Export a model to TensorRT")
# Step 3: Add arguments for trt_path and onnx_path
parser.add_argument("--trt_path", type=str, default=None, help="Path to save the TensorRT engine")
parser.add_argument("--onnx_path", type=str, default=None, help="Path to the ONNX model")
args = parser.parse_args()  # Step 4: Parse the arguments
print(f"args: {args}")

def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    # Step 5: Use values from parsed arguments if None
    trt_path = trt_path or args.trt_path
    onnx_path = onnx_path or args.onnx_path
    print(f"trt_path: {trt_path}, onnx_path: {onnx_path}")

    if trt_path is None:
        trt_path = input("Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input("Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
        input_profile=[
            {"input": [(1,3,256,256), (1,3,512,512), (1,3,1280,1280)]},
        ],
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret

# Step 6: Call export_trt without parameters
export_trt()
