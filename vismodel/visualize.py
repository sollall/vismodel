import matplotlib.pyplot as plt
import torch.nn as nn
from .utils import pickup_visualable_layers



"""
torch.nnモジュールで重みを持つ主なクラスは、torch.nn.Moduleを継承した以下のクラスです：
Linear（線形層）: torch.nn.Linear
Convolutional（畳み込み層）: torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d など
Recurrent（再帰層）: torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU など
Embedding（埋め込み層）: torch.nn.Embedding
Transformer（トランスフォーマー層）: torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer など
"""

def visualize_wandb(model):

    layers=pickup_visualable_layers(model)

    fig, axes = plt.subplots(4,len(layers), figsize=(15, 5))
    for i,(name,layer) in enumerate(layers):
        if isinstance(layer,nn.Linear):
            #weightも入力数変えたらだめなるのでは
            axes[0][i].imshow(layer.weight.detach().numpy(), cmap='viridis')
            axes[1][i].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap='viridis')
            #plt.colorbar()  # カラーバーを追加
        elif isinstance(layer,nn.Conv1d):
            weight_mat,bias_mat=layer.weight.unsqueeze(3),layer.bias.unsqueeze(1)
            weight_mat=weight_mat.view(weight_mat.size(0)*weight_mat.size(1)*weight_mat.size(2),weight_mat.size(3))
            axes[0][i].imshow(weight_mat.detach().numpy(), cmap='viridis')

            bias_mat=layer.bias.unsqueeze(1)
            axes[1][i].imshow(bias_mat.detach().numpy(), cmap='viridis')
            pass

        elif isinstance(layer,nn.Conv2d):
            weight_mat,bias_mat=layer.weight,layer.bias
            weight_mat=weight_mat.view(weight_mat.size(0)*weight_mat.size(1)*weight_mat.size(2),weight_mat.size(3))
            axes[0][i].imshow(weight_mat.detach().numpy(), cmap='viridis')

            bias_mat=layer.bias.unsqueeze(1)
            axes[1][i].imshow(bias_mat.detach().numpy(), cmap='viridis')
        elif isinstance(layer,nn.LSTM) or isinstance(layer,nn.RNN) or isinstance(layer,nn.GRU):
            #n_layersの数だけ表示させないとだけど今んとこ一層しか表示できてない
            #隠れ層のweightの次元は4*隠れ層の次元らしい
            for n in range(layer.num_layers):
                #lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0'
                axes[0][i].imshow(layer.weight_ih_l0.detach().numpy())
                axes[1][i].imshow(layer.bias_ih_l0.unsqueeze(1).detach().numpy())
                axes[2][i].imshow(layer.weight_hh_l0.detach().numpy())
                axes[3][i].imshow(layer.bias_hh_l0.unsqueeze(1).detach().numpy())
        else:
            print(name,layer)
