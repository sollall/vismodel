
def pickup_visualable_layers(model):
    """表示できるweightやbiasを持つlayerのみを持ってくる

    Args:
        model (_type_): _description_
    """
    visualable_layers=[]

    for name,layer in model.named_children():
        if hasattr(layer,"weight") or hasattr(layer,"bias"):
            visualable_layers.append((name,layer))
    return visualable_layers
