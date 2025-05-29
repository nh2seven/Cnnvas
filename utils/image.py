import torch
from torchvision import transforms
from PIL import Image
import yaml


# Class for image processing
class Img:
    def __init__(self, conf_path="config.yaml"):
        with open(conf_path, "r") as f:
            config = yaml.safe_load(f)

        self.size = config["data"]["image_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = config["data"]["mean"]
        self.std = config["data"]["std"]

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.postprocess = transforms.Compose(
            [
                transforms.Normalize(mean=[-m / s for m, s in zip(self.mean, self.std)], std=[1 / s for s in self.std]),
                transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
                transforms.ToPILImage(),
            ]
        )

    # Convert tensor to PIL image
    def tensor_to_pil(self, tensor):
        image = tensor.clone().detach().cpu().squeeze(0)
        return self.postprocess(image)

    # Convert PIL image to tensor
    def pil_to_tensor(self, pil_img):
        return self.preprocess(pil_img).unsqueeze(0).to(self.device)

    # Save image to file
    def save(self, tensor, path):
        pil_image = self.tensor_to_pil(tensor)
        pil_image.save(path)

    # Load image from file
    def load(self, path):
        image = Image.open(path).convert("RGB")
        return self.preprocess(image).unsqueeze(0).to(self.device)
