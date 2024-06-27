import argparse
import tensorrt as trt
import numpy as np

def build_engine(onnx_file_path, engine_file_path):
    # Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Create a builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse the ONNX model
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Create the builder config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Enable FP16 precision if available
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Build the engine
    print('Building the TensorRT engine...')
    engine = builder.build_engine(network, config)

    if engine is None:
        print('Failed to build the engine.')
        return None

    # Serialize the engine to a file
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f'Engine successfully saved to {engine_file_path}')
    return engine
def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--trt_path", type=str, required=True, help="Path to save the TensorRT engine.")
    args = parser.parse_args()

    build_engine(args.onnx_path, args.trt_path)

if __name__ == "__main__":
    main()