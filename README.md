# 🏛️ Historical Architecture GAN

A deep learning model that transforms modern architecture into historical styles using Generative Adversarial Networks (GANs) with memory-based reconstruction.

## 📜 Overview

This project implements a GAN-based approach to transform contemporary architectural images into their historical counterparts. Using structure-preserving attention mechanisms and memory networks, the model maintains the spatial layout and key architectural elements while transforming the style to reflect historical aesthetics.

## 🔍 Features

- **Structure-preserving transformation**: Maintains building proportions and architectural elements
- **Memory-based reconstruction**: Learns patterns from historical architecture pairs
- **Attention mechanisms**: Focuses on key structural elements during transformation
- **Skip connections**: Preserves important details from input images
- **Pretrained feature extraction**: Leverages architectural knowledge from pre-trained networks

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/historical-architecture-gan.git
cd historical-architecture-gan

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dataset

The model requires paired images of modern and historical architecture:
- Modern buildings (CURRENT)
- Historical versions/equivalents (PAST)

### Download

Download the prepared dataset from our Google Drive:
[Download Dataset](https://drive.google.com/drive/folders/1BT5vCdfN1FWRb-UKtvYhK0QONJW1bUUI?usp=drive_link)

Extract the contents into a `data` directory in your project folder.

## 🏃‍♀️ Training

```bash
python train.py
```

You can modify training parameters in `train.py`:
- Epochs
- Batch size
- Learning rate
- Loss weights

Checkpoints will be saved after each epoch in the project directory.

## 🧪 Inference

To transform a modern building image:

```bash
python main.py --input path/to/modern/building.jpg --output historical_result.jpg --checkpoint checkpoint_epoch_200.pth
```

## 📊 Results
Examples of transformation:
| Modern Building | Historical Transformation |
|-----------------|---------------------------|
| <img src="https://github.com/phamthangph13/Landscape-Archives/blob/main/Input/1.jpg" width="300"/> | <img src="https://github.com/phamthangph13/Landscape-Archives/blob/main/Result/1.jpg" width="300"/> |
| <img src="https://github.com/phamthangph13/Landscape-Archives/blob/main/Input/2.jpg" width="300"/> | <img src="https://github.com/phamthangph13/Landscape-Archives/blob/main/Result/2.jpg" width="300"/> |

## 📋 Model Architecture

### Generator
- **Encoder**: Extracts features with progressively increasing receptive field
- **Memory Network**: Embeds structural patterns with self-attention
- **Decoder**: Reconstructs the image with skip connections from the encoder

### Discriminator
- Structure-aware convolutional layers
- Self-attention modules for architectural coherence
- Patch-based style and structure discrimination

## ⚙️ Pretrained Checkpoints

Download our pretrained model (200 epochs) from:
[Download Checkpoints](https://mega.nz/folder/y2gVnbzA#QhJkC3Y85BFMMwgxy_OvNw)

## 📝 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.2.0
numpy>=1.19.5
matplotlib>=3.3.4
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Thanks to all contributors who provided architectural image pairs
- Special thanks to the computer vision community for GAN advancements
