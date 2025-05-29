import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# Function to compute the Gram matrix of a set of features
def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))

    return gram / (C * H * W)  # Normalized by the number of elements


# Class for Style Transfer implementation
class StyleTransfer:
    def __init__(self, model, content_layers=None, style_layers=None, conf_path="config.yaml"):
        # Load config for default parameters
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)
        
        self.model = model.eval()
        self.content_layers = content_layers or config["model"]["content_layers"]
        self.style_layers = style_layers or config["model"]["style_layers"]
        self.config = config

        # Freeze VGG parameters
        for param in self.model.parameters():
            param.requires_grad = False

    # Function to extract intermediate features from specific layers
    def ext_ft(self, x):
        features = self.model(x)

        content_ft = []
        style_ft = []

        for layer_name in self.content_layers:
            if layer_name in features:
                content_ft.append(features[layer_name])

        for layer_name in self.style_layers:
            if layer_name in features:
                style_ft.append(features[layer_name])

        return content_ft, style_ft

    # Function to compute losses for style transfer
    def loss(self, gen, content, style, alpha=None, beta=None):
        # Use config defaults if not provided
        alpha = alpha if alpha is not None else self.config["style_transfer"]["alpha"]
        beta = beta if beta is not None else self.config["style_transfer"]["beta"]
        
        gen_content, gen_style = self.ext_ft(gen)
        tgt_content, _ = self.ext_ft(content)
        _, tgt_style = self.ext_ft(style)

        content_loss = torch.tensor(0.0, device=gen.device, requires_grad=True)
        style_loss = torch.tensor(0.0, device=gen.device, requires_grad=True)

        # Calculate content losses
        for gc, tc in zip(gen_content, tgt_content):
            content_loss = content_loss + F.mse_loss(gc, tc)

        # Calculate style losses
        for gs, ts in zip(gen_style, tgt_style):
            gm_g = gram_matrix(gs)
            gm_t = gram_matrix(ts)
            style_loss = style_loss + F.mse_loss(gm_g, gm_t)

        total_loss = alpha * content_loss + beta * style_loss

        return total_loss, content_loss.item(), style_loss.item()
