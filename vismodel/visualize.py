import matplotlib.pyplot as plt
from utils import pickup_visualable_layers


def visualize_wandb(model):

    layers=pickup_visualable_layers(model)

    fig, axes = plt.subplots(2,len(layers), figsize=(15, 5))
    for i,(name,layer) in enumerate(layers):
        if isinstance(layer,nn.Linear):
            #weightも入力数変えたらだめなるのでは
            axes[0][i].imshow(layer.weight.detach().numpy(), cmap='viridis')
            axes[1][i].imshow(layer.bias.unsqueeze(1).detach().numpy(), cmap='viridis')
            #plt.colorbar()  # カラーバーを追加
        else:
            print(name,layer)
