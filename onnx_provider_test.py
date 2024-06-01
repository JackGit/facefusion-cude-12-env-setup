import onnxruntime as ort

# 获取可用的执行提供者
providers = ort.get_available_providers()
print("Available providers:", providers)

# 验证是否安装了 CUDA 提供者
if 'CUDAExecutionProvider' in providers:
    print("CUDAExecutionProvider is available.")
else:
    print("CUDAExecutionProvider is not available.")
