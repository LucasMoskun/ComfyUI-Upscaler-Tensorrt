import argparse
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_file_path, engine_file_path, device_id=0):
    # Set the GPU device context
    cuda.Device(device_id).make_context()
    
    # Create a TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create a builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse the ONNX file
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build the engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)
    
    # Save the engine to a file
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    # Pop the context to clean up
    cuda.Context.pop()
    
    return engine

def main():
    parser = argparse.ArgumentParser(description="Build a TensorRT engine from an ONNX model.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--trt_path", type=str, required=True, help="Path to save the TensorRT engine.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to use for engine building.")
    args = parser.parse_args()

    build_engine(args.onnx_path, args.trt_path, args.device_id)

if __name__ == "__main__":
    main()