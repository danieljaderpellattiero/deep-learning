import torch, platform
import tensorflow as tf

if __name__ == "__main__":
		print("Platform:", platform.platform())
		print("Python:", platform.python_version())
		print("CUDA compute capability:", torch.cuda.get_device_properties(0).major)
		print("PyTorch", torch.__version__, "| CUDA found ->", torch.cuda.is_available())
		print("PyTorch device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
		print("TensorFlow device:", tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else "CPU")
