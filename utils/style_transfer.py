import torch


# Function to compute the Gram matrix of a set of features
def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))

    return gram / (C * H * W)  # Normalized by the number of elements
