import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import torchdiffeq
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apng import APNG
from math import cos, pi, sin
from PIL import Image, ImageDraw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ニューラルネットワークの定義
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # ネットワーク中の線形変換の定義
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,hidden_size)
        self.l3 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        # ネットワーク中の具体的な計算
        x = self.l1(x)
        # 活性化関数．tanh, sigmoid, relu など
        x = torch.tanh(x)
        #x = torch.sigmoid(x)
        #x = torch.relu(x)
        x = self.l2(x)
        x = torch.tanh(x)
        x = self.l3(x)
        return x

    # 運動方程式の中では，ニューラルネットワークで表されたエネルギー関数の微分を使うため，もう一度，微分する．
    def grad(self, x):
        x = x.requires_grad_(True)
        with torch.enable_grad():
            hamiltonian = self(x)
            gradient = torch.autograd.grad(hamiltonian.sum(), x, create_graph=True, retain_graph=True)
        # 運動方程式の右辺．エネルギーの微分に St をかける．
        return torch.matmul(gradient[0], St)

    # 学習後のシミュレーション用の関数
    def fvec(self, t, x):
        return self.grad(x)


# csv ファイルに保存したデータを読み込む
# 今回は，データを上で作ったばかりなので，あまり意味はありません．
# 次回以降，自分で用意したデータを読み込むためのプログラミングの練習と考えてください．
# 自分で用意したデータを読み込むためには，以下の命令を使います．
# files.upload()
MY_BATCH_SIZE = 100

dftarget = pd.read_csv("target.csv", header=None, dtype=np.float32)
dfinput = pd.read_csv("input.csv", header=None, dtype=np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(dfinput.values, dftarget.values,test_size=0.2)

# 学習用データ
data_train = data_utils.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
train_loader = torch.utils.data.DataLoader(data_train,batch_size=MY_BATCH_SIZE,shuffle=True)

# テスト用データ
data_test = data_utils.TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))
test_loader = torch.utils.data.DataLoader(data_test,batch_size=MY_BATCH_SIZE,shuffle=True)

# 運動方程式を記述するための変数の定義
N = 1
# 行列　S を表すテンソルをつくる．
O = np.zeros((N,N))
Id = np.eye(N)
S = np.vstack([np.hstack([O, Id]), np.hstack([-Id, O])])
St = torch.tensor(-S, dtype=torch.float32).to(device)

# ネットワークの生成
input_size = 2
output_size = 1
hidden_size = 32
mynet = MLP(input_size,hidden_size,output_size).to(device)

# 学習回数の設定
num_epochs = 3000

# 誤差の計算方法を指定．
criterion = nn.MSELoss()

# 学習アルゴリズムとパラメータを指定． lr の部分を変えてみましょう．
optimizer = optim.Adam(params=mynet.parameters(), lr=0.001)

history_loss = []
history_eval = []
history_acc = []

# 実際の学習部分．
for epoch in range(num_epochs):
  # まず，ネットワークを学習モードに切り替える．
  mynet.train()

  total_loss = 0.0
  eval_loss = 0.0
  for i, (data, target) in enumerate(train_loader):
    # 微分をゼロに初期化．ネットワークの出力，損失関数を計算し，その微分を求め，学習を進める．
    optimizer.zero_grad()
    output = mynet.grad(data.to(device))

    loss = criterion(output, target.to(device))
    loss.backward()
    optimizer.step()

    # 各バッチでの損失関数の値を合計．
    total_loss = total_loss + loss.cpu().item()

  # （学習データとは別の）テストデータで性能を検証．
  num_correct = 0
  num_data = 0
  # 性能を検証するときには学習する必要は無いので，ネットワークを評価モードにする．
  mynet.eval()
  eval_loss = 0.0
  for i, (data, target) in enumerate(test_loader):
    output = mynet.grad(data.to(device))
    eval_loss = eval_loss + criterion(output, target.to(device)).cpu().item()

  history_loss.append(total_loss)
  history_eval.append(eval_loss)
  print("{}/{} training loss: {}, evaluation loss: {}".format(epoch+1,num_epochs,total_loss, eval_loss))

# 初期条件 1.0, 0.0 は，ぞれぞれ，はじめの位置と速度．
# いろいろと変えてみましょう．
x0 = torch.tensor([1.0, 0.0]).to(device)

# 0.0 から 10.0 までを100分割した時刻について，
# 解の計算を行う．
teval = torch.linspace(0.0, 10.0, 100).to(device)

# torchdiffeq.odeint に
# 　解くべき微分方程式（mynet の中の fvec という関数）
# 　初期条件 x0
# 　どの時刻での解を求めるか
# の3つを渡すと微分方程式を解いてくれる．
sol_model = torchdiffeq.odeint(mynet.fvec,x0,teval)

# 解をプロット．データを cpu に持ってきて numpy で扱える型に変換．
res = sol_model.detach().cpu().numpy()
teval = teval.detach().cpu().numpy()
plt.plot(teval,res[:,0])
plt.plot(teval,res[:,1])
plt.savefig("solution.png")

def make_animation(index, qval, pval):
    filename = "{:0>4}.png".format(index)
    im = Image.new("RGB", (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    x = 30*sin(qval) + 50
    y = 30*cos(qval) + 50
    draw.line((50, 50, x, y), fill=(0, 255, 0), width=2)
    draw.ellipse((x-5, y-5, x+5, y+5), fill=(0, 0, 255))
    im.save(filename)
    return filename

os.makedirs("./animation/", exist_ok=True)

files = []
for i in range(100):
    files.append(make_animation(i, res[i,0], res[i,1]))
APNG.from_files(files, delay=50).save("animation/animation_model.png")