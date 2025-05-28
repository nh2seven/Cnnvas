import torch
import torch.optim as optim
from models.vgg import VGG_FE
from utils.image import Img
from utils.style_transfer import StyleTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Main function to run the style transfer process
def main():
    vgg = VGG_FE().to(device)
    img = Img()

    # Load content and style images
    content_image = img.load("assets/content.jpg")
    style_image = img.load("assets/style.jpg")

    content_layers = ["conv4_2"]
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]

    style_transfer = StyleTransfer(vgg, content_layers, style_layers)
    generated_image = torch.randn_like(content_image).to(device).requires_grad_(True)


    optimizer = optim.Adam([generated_image], lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        loss, content_loss, style_loss = style_transfer.loss(generated_image, content_image, style_image, alpha=1.0, beta=1e3)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/1000], Loss: {loss.item():.4f}, Content: {content_loss:.4f}, Style: {style_loss:.4f}")

    img.save(generated_image, "output/generated.jpg")


if __name__ == "__main__":
    main()
