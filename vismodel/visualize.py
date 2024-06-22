from pathlib import Path

import matplotlib.pyplot as plt
import torch
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
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(layer.weight.detach().numpy(), cmap="viridis")
    axes[1].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap="viridis")

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_LayerNorm(name, layer):
    # LayerNormは次元が変わったりするのでめんどい
    # bias is 1d949428im?
    if layer.weight.dim() == 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(layer.weight.detach().numpy(), cmap="viridis")
        axes[1].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap="viridis")

    elif layer.weight.dim() == 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(layer.weight.detach().numpy(), cmap="viridis")
        axes[1].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap="viridis")
    else:
        raise Exception("unsport type in LayerNorm")

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


def make_image_Conv(matrix):
    """4次元のConvをいい感じに2次元にする

    Args:
        matrix (_type_): _description_

    Returns:
        _type_: _description_
    """
    bigmap = torch.empty(0)
    for i in range(matrix.shape[0]):
        small = torch.cat(list(matrix[i]), dim=1)
        bigmap = torch.cat([bigmap, small])

    return bigmap


def make_image_Conv2d(name, layer):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    weight_mat, bias_mat = layer.weight, layer.bias
    print(weight_mat.shape, bias_mat.shape)
    weight_reshape = make_image_Conv(weight_mat)
    axes[0].imshow(weight_reshape.detach().numpy(), cmap="viridis")
    for i in range(weight_reshape.shape[0] + 1)[:: weight_mat.shape[2]]:
        axes[0].axhline(y=i - 0.5, color="white", linewidth=2)
    for i in range(weight_reshape.shape[1] + 1)[:: weight_mat.shape[3]]:
        axes[0].axvline(x=i - 0.5, color="white", linewidth=2)

    bias_reshape = bias_mat.unsqueeze(1)
    axes[1].imshow(bias_reshape.detach().numpy(), cmap="viridis")
    for i in range(bias_reshape.shape[0] + 1)[:: bias_reshape.shape[0]]:
        axes[1].axhline(y=i - 0.5, color="white", linewidth=2)
    for i in range(bias_reshape.shape[1] + 1)[:: bias_reshape.shape[1]]:
        axes[1].axvline(x=i - 0.5, color="white", linewidth=2)

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_recurrent(name, layer):
    fig, axes = plt.subplots(2, 2, figsize=(15, 5))

    for n in range(layer.num_layers):
        # lstm.weight_ih_l0', 'lstm.weight_hh_l0', 'lstm.bias_ih_l0', 'lstm.bias_hh_l0'
        axes[0][0].imshow(layer.weight_ih_l0.detach().numpy().T)
        axes[0][1].imshow(layer.bias_ih_l0.unsqueeze(0).detach().numpy())
        axes[1][0].imshow(layer.weight_hh_l0.detach().numpy().T)
        axes[1][1].imshow(layer.bias_hh_l0.unsqueeze(0).detach().numpy())

    save_path = outputs_dir / name
    plt.savefig(str(save_path))

    return


def make_image_transformer(name, transformer):
    """
    Transfomrmer用にimageを作るのはあり layout色々変えたくなると思うから
    Transformer(
      (encoder): TransformerEncoder(
        (layers): ModuleList(
          (0-11): 12 x TransformerEncoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (linear1): Linear(in_features=512, out_features=2048, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=2048, out_features=512, bias=True)
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
          )
        )
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
      (decoder): TransformerDecoder(
        (layers): ModuleList(
          (0-5): 6 x TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (multihead_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
            )
            (linear1): Linear(in_features=512, out_features=2048, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
            (linear2): Linear(in_features=2048, out_features=512, bias=True)
            (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
            (dropout1): Dropout(p=0.1, inplace=False)
            (dropout2): Dropout(p=0.1, inplace=False)
            (dropout3): Dropout(p=0.1, inplace=False)
          )
        )
        (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      )
    )
    """
    # enocder
    # 1-6個
    for i, layer in enumerate(transformer.encoder.layers):
        visualize_wandb(layer)
    # self_attn
    # liner1
    # liner2
    # norm1
    # norm2

    # decoder
    for i, layer in enumerate(transformer.decoder.layers):
        visualize_wandb(layer)

    return


def visualize_wandb(model):
    # 各layerの処理を関数化すべきでは

    layers = pickup_visualable_layers(model.named_children())

    # layerごとに画像が作成されていくべきでは？
    for i, (name, layer) in enumerate(layers):
        if isinstance(layer, nn.Linear):  # or isinstance(layer, nn.LayerNorm):
            # weightも入力数変えたらだめなるのでは
            make_imgage_Linear(name, layer)

        elif isinstance(layer, nn.Conv1d):
            # unsqueezeの処理が違うだけなのでそこをここで直して可視化は共通処理にするべきだ
            make_image_Conv1d(name, layer)

        elif isinstance(layer, nn.Conv2d):
            make_image_Conv2d(name, layer)

        elif isinstance(layer, nn.LSTM) or isinstance(layer, nn.RNN) or isinstance(layer, nn.GRU):
            make_image_recurrent(name, layer)

        elif isinstance(layer, nn.Transformer):
            visualize_wandb(name, layer)
        else:
            print(name, layer)
