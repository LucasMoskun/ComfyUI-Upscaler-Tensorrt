import argparse
import tensorrt as trt

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

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision if available

    # Set the maximum workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1G

    # Build the serialized network
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print('Failed to build the engine.')
        return None

    # Deserialize the engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

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
