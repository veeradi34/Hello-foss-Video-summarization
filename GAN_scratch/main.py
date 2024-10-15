import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch, C, width, height = x.size()
        query = self.query(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, width * height)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        value = self.value(x).view(batch, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, width, height)
        
        return self.gamma * out + x

class ResidualGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_g):
        super(ResidualGenerator, self).__init__()
        self.init_size = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, feature_g * 8 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(feature_g * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_g * 8, feature_g * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_g * 4, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(feature_g * 4),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(feature_g * 4, feature_g * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(feature_g * 2, 0.8),
            nn.ReLU(inplace=True),
            SelfAttention(feature_g * 2),
            nn.Conv2d(feature_g * 2, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class ResidualDiscriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(ResidualDiscriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 3, 2, 1))]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(img_channels, feature_d, bn=False),
            *discriminator_block(feature_d, feature_d * 2),
            *discriminator_block(feature_d * 2, feature_d * 4),
            *discriminator_block(feature_d * 4, feature_d * 8),
            nn.Conv2d(feature_d * 8, 1, 1, stride=1, padding=0)
        )
        
    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1)

class MNISTUbyteDataset(Dataset):
    def __init__(self, images_path, labels_path, transform=None):
        self.images = self.load_images(images_path)
        self.labels = self.load_labels(labels_path)
        self.transform = transform

    def load_images(self, path):
        with open(path, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), np.uint8).astype(np.float32)
            data = data.reshape(-1, 28, 28)
            return data / 255.0

    def load_labels(self, path):
        with open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)

        return image, label

def evaluate(generator, latent_dim, device):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        gen_imgs = generator(z)
        vutils.save_image(gen_imgs, "evaluation_images.png", normalize=True)
    generator.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_channels = 1
feature_g = 64
feature_d = 64
batch_size = 128
epochs = 50
lr = 0.0002
beta1 = 0.5

generator = ResidualGenerator(latent_dim, img_channels, feature_g).to(device)
discriminator = ResidualDiscriminator(img_channels, feature_d).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

transform = transforms.Compose([
    transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

images_path = 'Path to your Dataset'
labels_path = 'Path to your Dataset'

dataset = MNISTUbyteDataset(images_path, labels_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        real = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        optimizer_d.zero_grad()
        real_validity = discriminator(imgs)
        real = torch.ones_like(real_validity).to(device)
        real_loss = criterion(real_validity, real)
        
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_imgs = generator(z)
        fake_validity = discriminator(gen_imgs.detach())
        fake = torch.zeros_like(fake_validity).to(device)
        fake_loss = criterion(fake_validity, fake)
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()
        fake_validity = discriminator(gen_imgs)
        real = torch.ones_like(fake_validity).to(device)
        g_loss = criterion(fake_validity, real)
        g_loss.backward()
        optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    vutils.save_image(gen_imgs, f"generated_epoch_{epoch+1}.png", normalize=True)
    evaluate(generator, latent_dim, device)
