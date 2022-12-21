# neural_network_study

# piano_by_FVM_VI

## 環境構築

WSLのインストール（省略）

WSLでGPUを使えるようにする（https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl 参照）

CUDAのバージョンによってはjaxが動かない可能性あり，適宜バージョンの変更を（ここではcuda-11）（https://www.nemotos.net/?p=2374 参照）

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