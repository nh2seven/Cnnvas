import torch
import torch.optim as optim
import yaml
from models.vgg import VGG_FE
from utils.image import Img
from utils.style_transfer import StyleTransfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Core style transfer logic
def st_core(content_tensor, style_tensor, config_path="config.yaml", epochs=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    vgg = VGG_FE().to(device)
    content_layers = config["model"]["content_layers"]
    style_layers = config["model"]["style_layers"]

    style_transfer = StyleTransfer(vgg, content_layers, style_layers, config_path)
    generated_image = torch.randn_like(content_tensor).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam([generated_image], lr=config["style_transfer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    total_epochs = epochs if epochs is not None else config["style_transfer"]["epochs"]
    print_interval = config["style_transfer"]["print_interval"]

    for epoch in range(total_epochs):
        optimizer.zero_grad()
        loss, content_loss, style_loss = style_transfer.loss(generated_image, content_tensor, style_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_([generated_image], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            generated_image.clamp_(-2.5, 2.5)

        if epoch % print_interval == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch}/{total_epochs}], Loss: {loss.item():.4f}, Content: {content_loss:.4f}, Style: {style_loss:.4f}, LR: {current_lr:.5f}")

    return generated_image


# Web interface function
def run_st(content_pil, style_pil, config_path="config.yaml", epochs=None):
    img_processor = Img(config_path)

    content_tensor = img_processor.pil_to_tensor(content_pil)
    style_tensor = img_processor.pil_to_tensor(style_pil)

    result_tensor = st_core(content_tensor, style_tensor, config_path, epochs)
    result_pil = img_processor.tensor_to_pil(result_tensor)

    return result_pil


# CLI function
def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_processor = Img(config_path)

    content_tensor = img_processor.load(config["paths"]["content_image"])
    style_tensor = img_processor.load(config["paths"]["style_image"])

    result_tensor = st_core(content_tensor, style_tensor, config_path)

    img_processor.save(result_tensor, config["paths"]["output_image"])
    print(f"Result saved to {config['paths']['output_image']}")


if __name__ == "__main__":
    main()
