import matplotlib.pyplot as plt
import numpy as np

def pickup_visualable_layers(layers):
    """表示できるweightやbiasを持つlayerのみを持ってくる

    Args:
        layers (dict): model.named_children()で呼び出せるlayersの情報
    """
    visualable_layers=[]

    for name,layer in layers:
        if hasattr(layer,"weight") or hasattr(layer,"bias"):
            visualable_layers.append((name,layer))
    return visualable_layers

def image_show(img):
    """cifarのimageを見れるようにする モデルの可視化という本来の趣旨とは関係ない

    Args:
        img (_type_): _description_
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def unite_weight_conv2d(weight_conv2d):
    """nn.Conv2dの重みを結合する 縦方向がoutput横軸がカーネル数

    Args:
        weight_conv2d (_type_): _description_
    """

    return weight_conv2d
