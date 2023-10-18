import torch

data_mean_std = {
    "default": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    "coco": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    "cub": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    # "cub": (
    #     torch.tensor([0.486, 0.499, 0.432]),
    #     torch.tensor([0.182, 0.181, 0.193]),
    # ),
    "car": (
        torch.tensor([0.471, 0.46, 0.455]),
        torch.tensor([0.267, 0.266, 0.271]),
    ),
    "craft": (
        torch.tensor([0.481, 0.512, 0.536]),
        torch.tensor([0.197, 0.195, 0.217]),
    ),
    "dog": (
        torch.tensor([0.475, 0.438, 0.383]),
        torch.tensor([0.23, 0.225, 0.221]),
    ),
    "nabird": (
        torch.tensor([0.491, 0.508, 0.464]),
        torch.tensor([0.168, 0.17, 0.187]),
    ),
    # "butterfly": (
    #     torch.tensor([0.63, 0.706, 0.555]),
    #     torch.tensor([0.168, 0.216, 0.205]),
    # ),
    "butterfly": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    "mel_erato": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    "bird525": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),
    "pet": (
        torch.tensor([0.485, 0.456, 0.406]),
        torch.tensor([0.229, 0.224, 0.225]),
    ),

}