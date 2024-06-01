import tensorflow as tf
import numpy as np
import tf2onnx
import onnx
import onnxruntime as ort

# 定义一个简单的卷积神经网络
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 打印模型摘要
model.summary()

# 保存 TensorFlow 模型
model.save("simple_cnn_tf")

# 将 TensorFlow 模型转换为 ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32),)
output_path = "simple_cnn.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

# 确保 CUDA 可用
print("CUDA available:", ort.get_device() == 'GPU')

# 加载 ONNX 模型
model_path = 'simple_cnn.onnx'
session = ort.InferenceSession(model_path)

# 打印模型输入输出信息
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input Name:", input_name)
print("Output Name:", output_name)

# 创建示例输入数据
# 假设模型的输入形状是 (1, 224, 224, 3)
input_data = np.random.randn(1, 224, 224, 3).astype(np.float32)

# 运行模型推理
result = session.run([output_name], {input_name: input_data})

# 打印结果
print("Result:", result)

# 分析结果
# 假设模型输出的是10类的概率分布，结果为一个形状为(1, 10)的数组
predicted_class = np.argmax(result[0])
predicted_probability = np.max(result[0])
print(f"Predicted class: {predicted_class} with probability {predicted_probability:.4f}")
