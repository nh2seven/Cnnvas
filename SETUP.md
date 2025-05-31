# Installing the NVIDIA Container Toolkit
In order to use GPU acceleration inside containers, there needs to be a bridge that allows Docker to communicate with the GPU drivers. Follow your distro's instructions to set this up:

## Fedora / RHEL / CentOS
```bash
# 1. Add the official repository
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo](https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo) | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2. Install the toolkit
sudo dnf install nvidia-container-toolkit

# 3. Configure the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker

# 4. Restart the Docker service to apply changes
sudo systemctl restart docker
```

## Ubuntu / Debian
```bash
# 1. Add the official repository key
curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2. Add the repository
curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3. Update package lists and install the toolkit
sudo apt-get update
sudo apt-get install nvidia-container-toolkit

# 4. Configure the Docker daemon
sudo nvidia-ctk runtime configure --runtime=docker

# 5. Restart the Docker service to apply changes
sudo systemctl restart docker
```