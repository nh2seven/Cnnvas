import torch
import torch.nn as nn
import torch.nn.functional as F


# Function to compute the Gram matrix of a set of features
def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))

    return gram / (C * H * W)  # Normalized by the number of elements


# Class for Style Transfer implementation
class StyleTransfer:
    def __init__(self, model, content, style):
        self.model = model.eval()
        self.content_layers = content
        self.style_layers = style

        # Freeze VGG parameters
        for param in self.model.parameters():
            param.requires_grad = False

    # Function to extract intermediate features from specific layers
    def ext_ft(self, x):
        content_ft = []
        style_ft = []

        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.content_layers:
                content_ft.append(x)
            if i in self.style_layers:
                style_ft.append(x)

        return content_ft, style_ft

    # Function to compute losses for style transfer
    def loss(self, gen, content, style, alpha=1.0, beta=1e5):
        gen_content, gen_style = self.extract_features(gen)
        tgt_content, _ = self.extract_features(content)
        _, tgt_style = self.extract_features(style)

        content_loss = 0.0
        style_loss = 0.0

        # Calculate content losses
        for gc, tc in zip(gen_content, tgt_content):
            content_loss += F.mse_loss(gc, tc)

        # Calculate style losses
        for gs, ts in zip(gen_style, tgt_style):
            gm_g = gram_matrix(gs)
            gm_t = gram_matrix(ts)
            style_loss += F.mse_loss(gm_g, gm_t)

        total_loss = alpha * content_loss + beta * style_loss

        return total_loss, content_loss.item(), style_loss.item()
