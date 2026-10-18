# 安装与搬迁

已有本地rdq环境时，直接在本工程中运行`python run.py doctor`即可。doctor会检查RDQ、detrex、detectron2源码以及detrex扩展来自本工程，不借用旧工程。

新机器建议Python3.10，CUDA驱动兼容PyTorch2.1.1+cu118：

```bash
conda create -n rdq-final python=3.10 -y
conda activate rdq-final
bash tools/install.sh
```

安装使用本地detrex/detectron2源码构建。需要与PyTorch匹配的CUDA toolkit、编译器和ninja；`MAX_JOBS=2`限制编译内存。工程携带的_C.so只在兼容的Python3.10/Linux/PyTorch/CUDA环境可直接使用，换环境应从所携带源码重新编译。CUDA驱动、PyTorch、Python包属于运行环境，工程源码不依赖旧项目目录。

requirements.txt为当前环境锁定版本，reference/environment_packages.json为整理时完整包清单。不要将旧项目目录添加到PYTHONPATH。统一入口会给子进程设置工程内部源码路径，并清除继承的RDQ_*路径覆盖；唯一外部数据位置由--data-root指定。

移动整个文件夹后在新位置执行`python run.py doctor`，内部路径由run.py所在位置计算；无需修改旧工程或建立指向旧工程的软链接。
