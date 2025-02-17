import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
from tqdm import tqdm

# Custom Dataset
class ArchitectureDataset(Dataset):
    def __init__(self, current_path, past_path, transform=None):
        self.current_images = [os.path.join(current_path, x) for x in os.listdir(current_path)]
        self.past_images = [os.path.join(past_path, x) for x in os.listdir(past_path)]
        self.transform = transform

    def __len__(self):
        return len(self.current_images)

    def __getitem__(self, idx):
        current_img = Image.open(self.current_images[idx]).convert('RGB')
        past_img = Image.open(self.past_images[idx]).convert('RGB')
        
        if self.transform:
            current_img = self.transform(current_img)
            past_img = self.transform(past_img)
        
        return current_img, past_img

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Memory Network
        self.memory_network = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.memory_network(x)
        x = self.decoder(x)
        return x

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=16, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train():
    print("Initializing training...")
    start_time = time.time()
    
    # Hyperparameters
    num_epochs = 200
    batch_size = 16
    lr = 0.0002
    beta1 = 0.5
    image_size = 256
    
    # Device configuration - kiểm tra và cấu hình GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU. Note: Training might be slow without GPU acceleration")
    
    print("\nSetting up data transformations and loading dataset...")
    # Transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset and DataLoader
    dataset = ArchitectureDataset(
        current_path='CURRENT',
        past_path='PAST',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset size: {len(dataset)} images")
    
    print("\nInitializing networks...")
    # Initialize networks
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    content_loss = nn.L1Loss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('CheckPoint', exist_ok=True)
    
    print("\nStarting training loop...")
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Progress bar for batches
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), 
                          desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, (current_imgs, past_imgs) in progress_bar:
            batch_start_time = time.time()
            
            batch_size = current_imgs.size(0)
            real_label = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_label = torch.zeros(batch_size, 1, 1, 1).to(device)
            
            # Move tensors to device
            current_imgs = current_imgs.to(device, non_blocking=True)
            past_imgs = past_imgs.to(device, non_blocking=True)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            d_real = discriminator(past_imgs)
            d_real_loss = adversarial_loss(d_real, real_label)
            
            fake_imgs = generator(current_imgs)
            d_fake = discriminator(fake_imgs.detach())
            d_fake_loss = adversarial_loss(d_fake, fake_label)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            d_fake = discriminator(fake_imgs)
            g_adversarial_loss = adversarial_loss(d_fake, real_label)
            g_content_loss = content_loss(fake_imgs, past_imgs)
            
            g_loss = g_adversarial_loss + 100 * g_content_loss
            g_loss.backward()
            g_optimizer.step()
            
            batch_time = time.time() - batch_start_time
            
            if (i + 1) % 100 == 0:
                progress_bar.set_postfix({
                    'd_loss': f'{d_loss.item():.4f}',
                    'g_loss': f'{g_loss.item():.4f}',
                    'batch_time': f'{batch_time:.2f}s'
                })
        
        epoch_time = time.time() - epoch_start_time
        print(f'\nEpoch {epoch+1} completed in {epoch_time:.2f} seconds')
        
        # Save models after each epoch
        print("Saving checkpoint...")
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }, os.path.join('CheckPoint', f'checkpoint_epoch_{epoch+1}.pth'))
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

if __name__ == '__main__':
    train()