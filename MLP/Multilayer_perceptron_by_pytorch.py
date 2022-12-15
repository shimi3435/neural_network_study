# pytorch 関係のライブラリの読み込み
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

# ニューラルネットワークの定義
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # ネットワーク中の線形変換の定義
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)
    
    def forward(self, x):
        # ネットワーク中の具体的な計算
        x = self.l1(x)
        # 活性化関数．tanh, sigmoid, relu など
        x = torch.tanh(x)
        #x = torch.sigmoid(x)
        #x = torch.relu(x)
        x = self.l2(x)
        return torch.softmax(x,dim=1)

if __name__ == "__main__":
    # 画像分類タスク用のデータの読み込みと整形．データを規格化してテンソル型に変形．
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # バッチサイズ．一度にいくつのデータを使用して学習を行うかを決定．
    # 小さいほど性能が上がるが，計算時間がかかる（並列計算時にデータが少なすぎて性能が出なくなるため）．
    MY_BATCH_SIZE = 100

    # データを利用できるように準備．
    train_loader = torch.utils.data.DataLoader(data_train,batch_size=MY_BATCH_SIZE,shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test,batch_size=MY_BATCH_SIZE,shuffle=False)

    for i in range(28):
        for j in range(28):
            print(f'{data_train.data[0][i][j]:4}', end='')
        print()
    plt.imshow(data_train.data[0], cmap='gray')
    print(data_train.targets[0].item())

    # 利用可能であれば GPU を使って計算．何か不具合が生じた場合は device = 'cpu' としましょう．
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    # ネットワークを生成．
    # ニューラルネットワークのサイズを決定．hidden の部分は変えても良いです．
    num_inputs = 784
    num_outputs = 10
    num_hidden = 32
    mynet = MLP(num_inputs,num_hidden,num_outputs).to(device)

    # どれだけ長く学習するか．はじめは小さい値で様子をみましょう．
    num_epochs = 20

    # 誤差の計算方法を指定．
    criterion = nn.CrossEntropyLoss()

    # 学習アルゴリズムとパラメータを指定． lr の部分を変えてみましょう．
    optimizer = optim.SGD(params=mynet.parameters(), lr=0.1)

    history_loss = []
    history_eval = []
    history_acc = []

    # 実際の学習部分．
    for epoch in range(num_epochs):
        # まず，ネットワークを学習モードに切り替える．
        mynet.train()

        total_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            # 画像を１次元配列に変換．
            data = data.view(-1, 28*28)

            # 微分をゼロに初期化．ネットワークの出力，損失関数を計算し，その微分を求め，学習を進める．
            optimizer.zero_grad()
            output = mynet(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()

            # 各バッチでの損失関数の値を合計．
            total_loss = total_loss + loss.cpu().item()

        # （学習データとは別の）テストデータで性能を検証．
        num_correct = 0
        num_data = 0
        # 性能を検証するときには学習する必要は無いので，ネットワークを評価モードにし，微分計算無しで計算．
        mynet.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for i, (data, target) in enumerate(test_loader):
                data = data.view(-1, 28*28)
                output = mynet(data.to(device))
                loss = criterion(output, target.to(device))
                eval_loss = total_loss + loss.cpu().item()
                num_correct = num_correct + output.cpu().argmax(dim=1).eq(target).sum()
                num_data = num_data + data.shape[0]
    
            history_loss.append(total_loss)
            history_eval.append(eval_loss)
            history_acc.append(num_correct.item()/num_data)
            print("{}/{} training loss: {}, evaluation loss: {}".format(epoch,num_epochs,total_loss,eval_loss))
            print("accuracy: {}/{}={}".format(num_correct, num_data,num_correct.item()/num_data))
            rnd=random.sample(range(len(target)),10)
            for i in range(10):
                print("(prediction: {}, truth: {}), ".format(output[rnd[i]].argmax().item(), target[rnd[i]].item()), end='')
                if(i==4 or i==9):
                    print()
            print()