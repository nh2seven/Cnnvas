import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary


# VGG Feature Extractor for Style Transfer
class VGG_FE(nn.Module):
    def __init__(self, content=None, style=None):
        super(VGG_FE, self).__init__()

        # Load pretrained VGG16 model
        vgg = models.vgg16(weights="IMAGENET1K_V1").features.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        # Select layers for content and style features
        self.content = content or ["conv4_2"]
        self.style = style or ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]
        self.required = set(self.content + self.style)

        # Define the model structure
        self.model = nn.Sequential()
        self.layers = {}
        conv_block = 1
        conv_in_block = 0
        layer_names = []

        # Iterate through the VGG layers and build the feature extractor
        for i, layer in enumerate(vgg.children()):
            if isinstance(layer, nn.Conv2d):
                conv_in_block += 1
                name = f"conv{conv_block}_{conv_in_block}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu{conv_block}_{conv_in_block}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool{conv_block}"
                conv_block += 1
                conv_in_block = 0
            else:
                name = f"layer{i}"

            self.model.add_module(name, layer)
            layer_names.append(name)
            self.layers[name] = len(self.model) - 1

            if name in self.required:
                continue
            if set(self.required).issubset(set(layer_names)):
                break

    def forward(self, x):
        features = {}
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.required:
                features[name] = x
        return features


# Wrapper class to summarize only the VGG layers
class SummaryWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model.model

    def forward(self, x):
        return self.model(x)


# Print model summary of the original VGG, minus the feature extraction layers
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = VGG_FE().to(device)
    summary_model = SummaryWrapper(base_model).to(device)

    print("\nModel Summary: (only VGG layers, no feature extraction)")
    summary(summary_model, input_size=(1, 3, 224, 224), device=device)
