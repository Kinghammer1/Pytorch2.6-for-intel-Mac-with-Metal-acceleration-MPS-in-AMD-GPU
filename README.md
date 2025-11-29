# Pytorch2.6-for-intel-Mac-with-Metal-acceleration-MPS-in-AMD-GPU
Pytorch2.6 for intel Mac with Metal acceleration on AMD GPU on Python==3.10
_________________________________________________________________
# 1️⃣Abstract/前言
1.Apple官方为intel芯片的Mac提供的Pytoch版本仅支持到Pytoch=2.2Version，https://developer.apple.com/metal/pytorch/ \
2.所以我创建了Pytorch2.6 for intel Mac with Metal acceleration on AMD GPU，以更好的为老款Mac提供MPS加速支持和更高版本的Pytorch和TorchVision \
3.如需whl版本，可以直接到Release下载，支持Python=3.10 and TorchVision=v0.21.0 \
4.来源：https://github.com/pytorch/pytorch
_________________________________________________________________
# 2️⃣Using Directly/直接使用
Python 3.10 Environment \
pip install torch-2.6.0a0+git1eba9b3-cp310-cp310-macosx_11_0_x86_64.whl \
pip install torchvision-0.21.0+7af6987-cp310-cp310-macosx_11_0_x86_64.whl \

_________________________________________________________________
# 3️⃣Methods/构建方法
***来自Deepseek，已经验证可行*** \
***如果需要直接使用，安装2️⃣Using Directly/直接使用自行安装即可***
## 环境准备

### 1. 清理环境并安装依赖
```bash
# 创建新的编译环境
conda create -n pytorch-build-2.6 python=3.10
conda activate pytorch-build-2.6

# 安装编译依赖
conda install cmake ninja numpy pyyaml mkl mkl-include setuptools cffi typing_extensions future six requests dataclasses
pip install -U pip

# 安装系统依赖
brew install cmake ninja git wget
brew install libomp
```

### 2. 确保 Xcode 工具链
```bash
# 检查 Xcode 版本
xcodebuild -version

# 确保命令行工具正确设置
sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
```

## 编译 PyTorch 2.6 并启用 MPS

### 1. 获取 PyTorch 2.6 源码
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0
git submodule sync
git submodule update --init --recursive

# 确保子模块正确更新
python -c "import os; os.system('git submodule status')"
```

### 2. 创建针对 Intel+AMD 优化的编译配置

创建编译脚本 `build_pytorch_2.6_mps.sh`：

```bash
#!/bin/bash

# 设置基础环境
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export MACOSX_DEPLOYMENT_TARGET=11.0

# 关键：启用 Metal/MPS 支持（基于 2.2.2 的成功配置）
export USE_MPS=1
export USE_METAL=1
export PYTORCH_ENABLE_MPS=1

# 添加 Metal 导出支持（这是 2.2.2 成功的关键）
export USE_PYTORCH_METAL_EXPORT=1

# 实验性：启用 AMD GPU 支持
export PYTORCH_ENABLE_MPS_AMD=1
export MPS_AMD_FORCE=1

# 禁用其他 GPU 后端
export USE_CUDA=0
export USE_CUDNN=0
export USE_NCCL=0
export USE_ROCM=0

# 优化 CPU 性能
export USE_MKLDNN=1
export USE_NNPACK=1
export USE_QNNPACK=1
export USE_PYTORCH_QNNPACK=1
export USE_XNNPACK=1

# 设置 Metal 框架路径
export METAL_LIBRARY_PATH="/System/Library/Frameworks/Metal.framework"
export METAL_SDK_PATH="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

# 并行编译
export MAX_JOBS=$(sysctl -n hw.ncpu)

echo "=== PyTorch 2.6 MPS 编译配置 ==="
echo "USE_MPS: $USE_MPS"
echo "USE_METAL: $USE_METAL"
echo "USE_PYTORCH_METAL_EXPORT: $USE_PYTORCH_METAL_EXPORT"
echo "METAL_LIBRARY_PATH: $METAL_LIBRARY_PATH"
echo "MAX_JOBS: $MAX_JOBS"

# 清理之前的构建
python setup.py clean

# 开始编译
python setup.py build develop
```

### 3. 应用针对 AMD GPU 的补丁

由于您使用的是 AMD GPU，可能需要一些调整：

```bash
# 在 pytorch 目录中创建补丁文件
cat > mps_amd_fix.patch << 'EOF'
--- a/cmake/Dependencies.cmake
+++ b/cmake/Dependencies.cmake
@@ -1234,6 +1234,12 @@ if(USE_METAL)
   if(NOT METAL_LIBRARY)
     message(WARNING "Metal library not found. Disabling Metal support.")
     set(USE_METAL OFF)
+  else()
+    message(STATUS "Found Metal library: ${METAL_LIBRARY}")
+    # 确保链接 Metal 框架
+    list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${METAL_LIBRARY})
+    # 添加 Metal Performance Shaders 框架
+    find_library(MPS_LIBRARY MetalPerformanceShaders)
+    if(MPS_LIBRARY)
+      list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${MPS_LIBRARY})
+    endif()
   endif()
 endif()
EOF

# 尝试应用补丁
git apply mps_amd_fix.patch || echo "补丁可能不完全适用，继续编译..."
```

### 4. 运行编译
```bash
# 给脚本执行权限
chmod +x build_pytorch_2.6_mps.sh

# 运行编译
./build_pytorch_2.6_mps.sh
```

## 替代编译方法（如果上述方法失败）

### 方法 B：使用 setup.py 直接编译
```bash
# 在 pytorch 目录中执行
python setup.py clean

# 使用 setup.py 直接配置
CMAKE_ARGS="-DUSE_MPS=ON -DUSE_METAL=ON -DUSE_PYTORCH_METAL_EXPORT=ON -DUSE_CUDA=OFF -DUSE_ROCM=OFF" \
python setup.py build develop
```

### 方法 C：分步 CMake 编译
```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake .. \
  -DUSE_MPS=ON \
  -DUSE_METAL=ON \
  -DUSE_PYTORCH_METAL_EXPORT=ON \
  -DUSE_CUDA=OFF \
  -DUSE_ROCM=OFF \
  -DUSE_MKLDNN=ON \
  -DUSE_NNPACK=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python) \
  -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} \
  -DMETAL_LIBRARY_PATH="/System/Library/Frameworks/Metal.framework"

# 编译
make -j$(sysctl -n hw.ncpu)

# 安装
cd ..
python setup.py develop
```

## 验证编译结果

创建验证脚本 `verify_mps_2.6.py`：

```python
import torch
import sys
import platform

print("=== PyTorch 2.6 MPS 验证 ===")
print(f"PyTorch version: {torch.__version__}")
print(f"Python: {sys.version}")
print(f"macOS: {platform.mac_ver()[0]}")
print(f"Architecture: {platform.machine()}")

print("\n=== 编译配置 ===")
print(f"Build settings: {torch.__config__.show()}")

print("\n=== MPS 支持检测 ===")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"MPS device: {device}")
    
    # 性能测试
    import time
    size = 3000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(3):
        _ = a @ b
    if hasattr(torch, 'mps'):
        torch.mps.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(10):
        c = a @ b
    if hasattr(torch, 'mps'):
        torch.mps.synchronize()
    mps_time = time.time() - start_time
    
    print(f"MPS 矩阵乘法时间: {mps_time:.4f}s")
    
    # 对比 CPU
    a_cpu, b_cpu = a.cpu(), b.cpu()
    start_time = time.time()
    for _ in range(10):
        c_cpu = a_cpu @ b_cpu
    cpu_time = time.time() - start_time
    
    print(f"CPU 矩阵乘法时间: {cpu_time:.4f}s")
    print(f"加速比: {cpu_time/mps_time:.2f}x")
    
    # 内存信息
    if hasattr(torch, 'mps'):
        try:
            current_mem = torch.mps.current_allocated_memory()
            driver_mem = torch.mps.driver_allocated_memory()
            print(f"MPS 当前内存: {current_mem/1024**2:.1f} MB")
            print(f"MPS 驱动内存: {driver_mem/1024**2:.1f} MB")
        except Exception as e:
            print(f"内存信息获取失败: {e}")
else:
    print("MPS 不可用")
    
print("\n=== 关键编译标志验证 ===")
# 检查是否包含 Metal 支持
build_string = str(torch.__config__.show())
key_flags = ['MPS', 'METAL', 'USE_PYTORCH_METAL_EXPORT']
for flag in key_flags:
    if flag in build_string:
        print(f"✅ {flag}: 已启用")
    else:
        print(f"❌ {flag}: 未找到")
```

## 故障排除

### 常见问题 1: Metal 库找不到
```bash
# 确保 Metal 框架路径正确
sudo xcode-select -switch /Applications/Xcode.app/Contents/Developer
export METAL_LIBRARY_PATH="/System/Library/Frameworks/Metal.framework"
```

### 常见问题 2: 链接错误
```bash
# 完全清理后重试
git clean -xdf
git submodule foreach --recursive git clean -xdf
python setup.py clean
```

### 常见问题 3: Python 包冲突
```bash
# 在干净的 conda 环境中编译
conda deactivate
conda env remove -n pytorch-build-2.6
conda create -n pytorch-build-2.6 python=3.10
conda activate pytorch-build-2.6
# 重新安装依赖...
```

### 常见问题 4: 子模块问题
```bash
# 强制更新所有子模块
git submodule deinit -f .
git submodule update --init --recursive
```

## 成功编译的标志

编译成功后，您应该在验证脚本中看到：
- ✅ `MPS available: True`
- ✅ `MPS built: True` 
- ✅ 在编译配置中包含 `USE_MPS`、`METAL` 等关键标志
- ✅ 能够创建 `device='mps'` 的张量
- ✅ 比 CPU 更快的计算速度

## 安装到其他环境

编译成功后，您可以创建 wheel 包安装到其他环境：
```bash
# 创建 wheel 包
python setup.py bdist_wheel

# 安装到目标环境
pip install dist/torch-2.6.0*.whl
```

_________________________________________________________________
# 4️⃣Test/测试脚本
Python \
import torch \
print(torch.__version__)  # 应为 2.6.0 \
print(torch.backends.mps.is_available())  # 应为 True 
