import tensorflow as tf
from tensorflow.python.client import device_lib

print("TensorFlow version:", tf.__version__)
print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
print("Available devices:")
for device in device_lib.list_local_devices():
    print(device)
