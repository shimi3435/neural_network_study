import os
from apng import APNG
import matplotlib.pyplot as plt
import autograd
import autograd.numpy as np
import scipy.integrate
solver = scipy.integrate.solve_ivp

import visualization_by_apng

# テスト用のデータの作成
# 自分で用意したデータを利用するときには不要．
N = 1
# 行列 S に摩擦などに関する項を追加する．
O = np.zeros((N,N))
Id = np.eye(N)
dId = -0.1*Id
S = np.vstack([np.hstack([O, Id]), np.hstack([-Id, dId])])

# 調和振動子のエネルギー
def energy(u):
  return 0.5*u[0]*u[0] + u[1]*u[1]

# 実験用データ作成のための右辺項
def odefunc(t,u):
  dhdu = autograd.grad(energy)(u)
  return np.matmul(S, dhdu)

flag = False
# 初期条件．時刻 0 での値を適当に決めます．
for idx in range(100):
    u = np.zeros(2)
    u = np.random.randn(2)

    # シミュレーションを行う時間区間を設定し，それを細かく分割．
    M = 50
    tend = 5.0
    t_eval = np.linspace(0, tend, M)
    # データの時間間隔を計算し，dt という名前で保存．
    dt = t_eval[1]-t_eval[0]

    # 実際に微分方程式の解を計算．
    # 解は solver の返り値の sol の中の 'y' という部分に保存される．
    sol = solver(odefunc,[0, tend], u,t_eval=t_eval)
    tval = sol['t']
    q, p = sol['y'][0], sol['y'][1]

    # Input: 対応する状態変数データの作成
    xtmp = np.stack([q, p]).T
    x1 = xtmp[:-1,:]
    x2 = xtmp[1:,:]
    input_data = (x1 + x2)/2.0
    # Target: 勾配データの作成（観測データから微分を推定するイメージで有限差分で作る）
    dudt = (sol['y'][:,1:]-sol['y'][:,:-1])/dt

    if flag:
        inlist = np.concatenate([inlist, input_data],axis=0)
        targetlist = np.concatenate([targetlist, dudt.T],axis=0)
    else:
        inlist=input_data
        targetlist=dudt.T
        flag=True

plt.plot(tval ,q, label="q")
plt.plot(tval ,p, label="p")
plt.legend()
plt.savefig("q_p.png")
    
np.savetxt("target.csv",targetlist,delimiter=',')
np.savetxt("input.csv",inlist,delimiter=',')

os.makedirs("./animation/", exist_ok=True)

files = []
for i in range(len(q)):
    files.append(visualization_by_apng.make_animation(i, q[i], p[i]))    
APNG.from_files(files, delay=50).save("./animation/animation_data.png")
