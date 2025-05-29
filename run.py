import torch
import torch.optim as optim
import yaml
from models.vgg import VGG_FE
from utils.image import Img
from utils.style_transfer import StyleTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Main function to run the style transfer process
def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vgg = VGG_FE().to(device)
    img = Img(config_path)

    # Load content and style images
    content_image = img.load(config["paths"]["content_image"])
    style_image = img.load(config["paths"]["style_image"])

    content_layers = config["model"]["content_layers"]
    style_layers = config["model"]["style_layers"]

    style_transfer = StyleTransfer(vgg, content_layers, style_layers, config_path)
    generated_image = torch.randn_like(content_image).to(device).requires_grad_(True)

    optimizer = optim.Adam([generated_image], lr=config["style_transfer"]["learning_rate"])

    epochs = config["style_transfer"]["epochs"]
    print_interval = config["style_transfer"]["print_interval"]

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, content_loss, style_loss = style_transfer.loss(generated_image, content_image, style_image)
        loss.backward()
        optimizer.step()

        if epoch % print_interval == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Content: {content_loss:.4f}, Style: {style_loss:.4f}")

    img.save(generated_image, config["paths"]["output_image"])
    print(f"Result saved to {config['paths']['output_image']}")


if __name__ == "__main__":
    main()
