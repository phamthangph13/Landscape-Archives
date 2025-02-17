import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from train import Generator

def load_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

def denormalize(tensor):
    tensor = tensor.clone().detach()
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    tensor = tensor.cpu().numpy()
    return np.transpose(tensor[0], (1, 2, 0))

def generate_historical_view(input_path, output_path, checkpoint_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize and load generator
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    # Load and process input image
    input_image = load_image(input_path).to(device)
    
    # Generate historical view
    with torch.no_grad():
        generated_image = generator(input_image)
    
    # Convert to numpy array and denormalize
    generated_image = denormalize(generated_image)
    
    # Save the generated image
    output_image = Image.fromarray((generated_image * 255).astype(np.uint8))
    output_image.save(output_path)

if __name__ == '__main__':
    input_path = 'test.jpg'  # Ảnh đầu vào
    checkpoint_dir = 'CheckPoint'  # Thư mục chứa các checkpoint
    output_dir = 'TEST'  # Thư mục lưu kết quả

    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Duyệt qua các checkpoint từ 1 -> 200
    for epoch in range(1, 201):
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        output_path = os.path.join(output_dir, f'generated_epoch_{epoch}.jpg')

        if os.path.exists(checkpoint_path):
            generate_historical_view(input_path, output_path, checkpoint_path)
            print(f"Saved: {output_path}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
