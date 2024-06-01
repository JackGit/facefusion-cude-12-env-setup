import tensorflow as tf

# 创建一个简单的 TensorFlow 操作以验证 GPU 是否工作
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print("Result of matrix multiplication:")
print(c)

# 打印可用的 GPU 设备
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
