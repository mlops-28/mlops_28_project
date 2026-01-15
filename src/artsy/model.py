from pytorch_lightning import LightningModule
from torch import nn, optim
import torch

class ArtsyClassifier(LightningModule):
    """CNN with 3 convolutional layers to classify the artstyle of 256x256 images."""
    def __init__(self, lr: float=1e-3, drop_p: float=0.2):
        super().__init__()
        # Out: floor((in + 2*padding - kernel_size) / stride) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1) # Out: 
        self.fc = nn.Linear(64*14*14, 5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        self.criterium = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x)) # Out: floor((256 + 0 - 5) / 2) + 1 = 126
        x = torch.max_pool2d(x, 2, 2) # Out: floor(126 / 2) = 63
        x = self.relu(self.conv2(x)) # Out: floor((63 + 0 - 3) / 1) + 1 = 61
        x = torch.max_pool2d(x, 2, 2) # Out: floor(63 / 2) = 31
        x = self.relu(self.conv3(x)) # Out: floor((31 + 0 - 3) / 1) + 1 = 29
        x = torch.max_pool2d(x, 2, 2) # Out: floor(29 / 2) = 14
        x = self.dropout(torch.flatten(x, 1)) # Out: 64 * 14 * 14 = 12544

        return self.fc(x)
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = ArtsyClassifier()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
