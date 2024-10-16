import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_g):
        super(WGANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class WGANDiscriminator(nn.Module):
    def __init__(self, img_channels, feature_d):
        super(WGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, feature_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 4, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        return self.model(img).view(-1, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
img_channels = 1
feature_g = 64
feature_d = 64
batch_size = 128
epochs = 50
lr = 0.0002

generator = WGANGenerator(latent_dim, img_channels, feature_g).to(device)
discriminator = WGANDiscriminator(img_channels, feature_d).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

images_path = 'Path to Dataset'
labels_path = 'Path to Dataset'

dataset = MNISTUbyteDataset(images_path, labels_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def gradient_penalty(disc, real, fake):
    batch_size, C, H, W = real.shape
    epsilon = torch.rand((batch_size, 1, 1, 1), device=device).expand_as(real)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    prob_interpolated = disc(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        batch_size = imgs.size(0)
        z = torch.randn(batch_size, latent_dim, 1, 1, device=device)

        optimizer_d.zero_grad()
        real_loss = discriminator(imgs).mean()
        fake_loss = discriminator(generator(z).detach()).mean()
        gp = gradient_penalty(discriminator, imgs, generator(z).detach())
        d_loss = -real_loss + fake_loss + 10 * gp
        d_loss.backward()
        optimizer_d.step()

        if i % 5 == 0:
            optimizer_g.zero_grad()
            g_loss = -discriminator(generator(z)).mean()
            g_loss.backward()
            optimizer_g.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    vutils.save_image(generator(z), f"generated_epoch_{epoch+1}.png", normalize=True)
