from pathlib import Path

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

outputs_dir = Path("outputs")
outputs_dir.mkdir(exist_ok=True)


def make_imgage_Linear(name, layer):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    axes[0].imshow(layer.weight.detach().numpy(), cmap="viridis")
    axes[1].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap="viridis")

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_Conv1d(name, layer):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5))
    weight_mat, bias_mat = layer.weight.unsqueeze(3), layer.bias.unsqueeze(1)
    weight_mat = weight_mat.view(weight_mat.size(0) * weight_mat.size(1) * weight_mat.size(2), weight_mat.size(3))
    axes[0].imshow(weight_mat.detach().numpy(), cmap="viridis")

    bias_mat = layer.bias.unsqueeze(1)
    axes[1].imshow(bias_mat.detach().numpy(), cmap="viridis")

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_Conv2d(name, layer):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5))

    weight_mat, bias_mat = layer.weight, layer.bias
    weight_mat = weight_mat.view(weight_mat.size(0) * weight_mat.size(1) * weight_mat.size(2), weight_mat.size(3))
    axes[0].imshow(weight_mat.detach().numpy(), cmap="viridis")

    bias_mat = layer.bias.unsqueeze(1)
    axes[1].imshow(bias_mat.detach().numpy(), cmap="viridis")

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_LSTM(name, layer):
    fig, axes = plt.subplots(4, 1, figsize=(15, 5))

    for n in range(layer.num_layers):
        # lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0'
        axes[0].imshow(layer.weight_ih_l0.detach().numpy())
        axes[1].imshow(layer.bias_ih_l0.unsqueeze(1).detach().numpy())
        axes[2].imshow(layer.weight_hh_l0.detach().numpy())
        axes[3].imshow(layer.bias_hh_l0.unsqueeze(1).detach().numpy())

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


class Vismodule(nn.Module):
    def __init__(self):
        super(Vismodule, self).__init__()

    def visualize_wandb(self):
        # 各layerの処理を関数化すべきでは

        layers = pickup_visualable_layers(self.named_children())

        # layerごとに画像が作成されていくべきでは？
        for i, (name, layer) in enumerate(layers):
            if isinstance(layer, nn.Linear):
                # weightも入力数変えたらだめなるのでは
                make_imgage_Linear(name, layer)

            elif isinstance(layer, nn.Conv1d):
                make_image_Conv1d(name, layer)

            elif isinstance(layer, nn.Conv2d):
                make_image_Conv2d(name, layer)

            elif isinstance(layer, nn.LSTM) or isinstance(layer, nn.RNN) or isinstance(layer, nn.GRU):
                make_image_LSTM(name, layer)
            else:
                print(name, layer)
