## install cuda
need to find out which version of cuda need to be installed by checking nvidia-smi(driver version)
```bash
nvidia-smi
```

for example it return version of 12.4, then need to pip install cuda of 12.4.


if cuda installed, check version by running install:
```bash
sudo apt-get install cuda 
```
output would like: 提示： cuda 已经是最新版 (12.5.0-1)。

or run `nvcc --version` to check version of cuda.

if version is not 12.4, remove old cuda lib:
```bash
sudo apt-get remove --purge cuda
sudo apt-get remove --purge "cublas*" "cufft*" "curand*" "cusolver*" "cusparse*" "npp*" "nvjpeg*"
sudo apt-get remove --purge "libcudnn*"
sudo apt-get autoremove
```

install cuda 12.4
```bash
sudo apt-get install cuda-12-4
```

now `nvcc --version` shows Cuda compilation tools, release 12.4, V12.4.131

## install tensorflow==2.15
for cuda 12, have to install tensorflow 2.15(2.16 won't be able to detect GPU, ref: https://github.com/tensorflow/tensorflow/issues/63341
)

check tensorflow version: 
```bash
pip show tensorflow
```

install 2.15 version:
```bash
pip install tensorflow==2.15
```

test if tensorflow can use GPU:
```bash
python tf_gpu_test.py
```

## install cudnn
https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

```bash
sudo apt-get -y install cudnn-cuda-12
```
cudnn9-cuda-12 已经是最新版 (9.1.1.17-1)。

verify cudnn installation:
```bash
python cudnn_test.py
```

## ONNX Runtime
for cuda 12.x, 
- ONNX should be 1.18 for 1.17
- cuDNN should be 8.9.2.26
（based on: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html）

install onnx by running `python install.py` of facefusion script.

test onnx with gpu:
```bash
python onnx_gpu_test.py
```

it should finish the job by using GPU.
```bash
CUDA available: True
Input Name: args_0
Output Name: dense_1
Result: [array([[0.07299842, 0.04760437, 0.31281087, 0.06697365, 0.05677573,
        0.13065308, 0.06251975, 0.13218783, 0.06670961, 0.05076663]],
      dtype=float32)]
```

### ONNX with GPU
but at this time, start facefusion by run.py with CUDA executor, video processing would emit error like:
```
/onnxruntime_src/onnxruntime/core/session/provider_bridge_ort.cc:1426 onnxruntime::Provider& onnxruntime::ProviderLibrary::Get() [ONNXRuntimeError] : 1 : FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.8: cannot open shared object file: No such file or directory
```

main issue is facefusion can't find onnx cuda provider. gpt says it would located at path like `/usr/local/cuda/lib64`
but it does't for my case

by running below script to locate provider file:
```bash
find $(python -c 'import site; print(site.getsitepackages()[0])') -name "libonnxruntime_providers_cuda.so"
```

output: /home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so

发现
/home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/onnxruntime/capi 
目录下有：
libonnxruntime_providers_cuda.so
libonnxruntime_providers_shared.so
libonnxruntime_providers_tensorrt.so

and by `sudo find /home/jack -name "libcudnn.so.8"` find out:
/home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.8

**need to add path to .zshrc**
```bash
export LD_LIBRARY_PATH=/home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/onnxruntime/capi:/home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/nvidia/cudnn/lib
```

check env by `echo $LD_LIBRARY_PATH`:
output: /home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/onnxruntime/capi:/home/jack/anaconda3/envs/facefusion/lib/python3.10/site-packages/nvidia/cudnn/lib

and now run the facefusion would fix missing onnx cuda so file issue.