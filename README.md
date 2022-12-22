# neural_network_study

# piano_by_FVM_VI

## 環境構築

WSLのインストール（省略）

WSLでGPUを使えるようにする（https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl 参照）

CUDAのバージョンによってはjax，pytorch，Tensorflowが動かない可能性あり，適宜バージョンの変更を（ここではcuda-11.8）（https://www.nemotos.net/?p=2374 参照）

pyenvのインストール（省略）

pythonの仮想環境の構築（ここでは3.10.4）
```
pyenv install 3.10.4
pyenv shell 3.10.4
python -m venv .venv
```

pythonのライブラリのインストール
```
pip install torch
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax
pip install tensorflow
pip install tensorflow_datasets
```

### エラー

#### その1：TensorflowでGPUが認識しない

```
Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda/lib64:
2022-12-21 16:22:48.240171: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda/lib64:/usr/local/cuda/lib64:
```

対処法

・まずTensorRTをインストールする（https://developer.nvidia.com/nvidia-tensorrt-8x-download参照）

```
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.5.1-cuda-11.8_1.0-1_amd64.deb #後ろは上記URLからDLしたもの
sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-8.5.1-cuda-11.8/nv-tensorrt-local-3E18D84E-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install tensorrt
```

・シンボリックリンクをはる

```
pip install tensorrt
find / -name libnvinfer.so.8
#/home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/libnvinfer.so.8
sudo ln -s /home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/libnvinfer.so.8 /home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/libnvinfer.so.7
sudo ln -s /home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.8 /home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/libnvinfer_plugin.so.7
```

・パスを通す

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shimi3435/workspace/Python3/neural_network_study/.venv/lib/python3.10/site-packages/tensorrt/
```

これでTensorflowでGPUが認識するようになった

#### その2：Cudnnのバージョンがあってない

```
2022-12-21 17:18:57.232955: E external/org_tensorflow/tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:421] Loaded runtime CuDNN library: 8.5.0 but source was compiled with: 8.6.0.  CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.
2022-12-21 17:18:57.238863: E external/org_tensorflow/tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:421] Loaded runtime CuDNN library: 8.5.0 but source was compiled with: 8.6.0.  CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.
2022-12-21 17:18:57.245042: E external/org_tensorflow/tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:421] Loaded runtime CuDNN library: 8.5.0 but source was compiled with: 8.6.0.  CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library.  If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.
```

対処法：Tensorflowのバージョンをさげる（2.11.0から2.9.0へ）

```
pip install --upgrade tensorflow==2.9.0
```