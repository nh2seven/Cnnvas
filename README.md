# Cnnvas
Cnnvas is a CNN-based neural style transfer (NST) tool for generating artistic images.

## Features
- Takes content and style images as input, produces stylized image that largely follows the concept of [Deep Art](https://en.wikipedia.org/wiki/DeepArt).
- Interactive web interface built with Gradio for simplified usage.
- Available as a container for easy reproducibility. 
- The Docker image is also available on [Docker Hub](https://hub.docker.com/r/nh2seven/cnnvas).

## Requirements
### Hardware Requirements
These requirements are exclusive of system usage (tested with a Nvidia GPU):
- **GPU VRAM**: 2GB minimum, 4GB recommended.
- **System RAM**: 1GB minimum, 2GB recommended.
- **Storage**: 10GB of free space (mostly for the Docker image).

### Software Requirements
Given the hardware and corresponding drivers, you'll need these to go with it:
1. **Docker**: Takes care of Python and PyTorch dependencies.
2. **NVIDIA Container Toolkit**: This allows Docker to access the host's GPU. Refer to [SETUP](SETUP.md) to learn how to set this up.

## Getting Started
### Option A: Pull from Docker Hub (Recommended)
Pull the image directly from Docker Hub:
```bash
docker pull nh2seven/cnnvas
```

### Option B: Build from Source
If you prefer to build the image yourself, follow these steps:
```bash
# 1. Clone this repository
git clone https://github.com/nh2seven/Cnnvas.git
cd Cnnvas

# 2. Build the container
docker build -t cnnvas .
```

## Running Cnnvas
### With GPU Support (Recommended)
This is the intended way of running `Cnnvas` locally:
```bash
# Use all available GPUs
docker run --gpus all -p 7860:7860 --name cnnvas --rm nh2seven/cnnvas

# Or, specify a particular GPU (0/1/...)
docker run --gpus device=0 -p 7860:7860 --name cnnvas --rm nh2seven/cnnvas
```

### With CPU Only
This is slow and will be excessively heavy on your CPU:
```bash
docker run -p 7860:7860 --rm --name cnnvas nh2seven/cnnvas
```

## Usage
- After starting the container, access the web interface at: [http://localhost:7860](http://localhost:7860)
- Upload the content and style images of your choice.
- Set the number of epochs to run. More epochs is better.
- After a while, a stylized image will be generated, available to download and save.

## Contributing
Contributions are welcome. Feel free to submit a pull request or open an issue, and I'll have a look at it!

## License
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.