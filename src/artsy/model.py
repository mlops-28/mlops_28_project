from pytorch_lightning import LightningModule
from torch import nn, optim
import torch

class ArtsyClassifier(LightningModule):
    """CNN with 3 convolutional layers to classify the artstyle of 256x256 images."""
    def __init__(self, lr: float=1e-3, drop_p: float=0.2) -> None:
        super().__init__()
        # Out: floor((in + 2*padding - kernel_size) / stride) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2) 
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1) # Out: 
        self.fc = nn.Linear(64*6*6, 5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        self.criterium = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x)) # Out: floor((128 + 0 - 5) / 2) + 1 = 62
        x = torch.max_pool2d(x, 2, 2) # Out: floor(62 / 2) = 31
        x = self.relu(self.conv2(x)) # Out: floor((31 + 0 - 3) / 1) + 1 = 29
        x = torch.max_pool2d(x, 2, 2) # Out: floor(29 / 2) = 14
        x = self.relu(self.conv3(x)) # Out: floor((14 + 0 - 3) / 1) + 1 = 12
        x = torch.max_pool2d(x, 2, 2) # Out: floor(12 / 2) = 6
        x = self.dropout(torch.flatten(x, 1)) # Out: 64 * 6 * 6 = 2304

        return self.fc(x)
    
    def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        self.half()
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        self.half()
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)

        self.log("val_loss", loss)

        return loss
    
    def test_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        self.half()
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)

        self.log("test_loss", loss)

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
