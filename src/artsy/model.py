from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
import torchmetrics


class ArtsyClassifier(LightningModule):
    """CNN with 3 convolutional layers to classify the artstyle of 256x256 images."""

    def __init__(self, lr: float = 1e-3, drop_p: float = 0.2) -> None:
        super().__init__()
        # Out: floor((in + 2*padding - kernel_size) / stride) + 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(64 * 6 * 6, 5)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)
        self.criterium = nn.CrossEntropyLoss()
        self.lr = lr
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

        self.label_map = {9: 0, 12: 1, 20: 2, 21: 3, 23: 4}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # Out: floor((128 + 0 - 5) / 2) + 1 = 62
        x = torch.max_pool2d(x, 2, 2)  # Out: floor(62 / 2) = 31
        x = self.relu(self.conv2(x))  # Out: floor((31 + 0 - 3) / 1) + 1 = 29
        x = torch.max_pool2d(x, 2, 2)  # Out: floor(29 / 2) = 14
        x = self.relu(self.conv3(x))  # Out: floor((14 + 0 - 3) / 1) + 1 = 12
        x = torch.max_pool2d(x, 2, 2)  # Out: floor(12 / 2) = 6
        x = self.dropout(torch.flatten(x, 1))  # Out: 64 * 6 * 6 = 2304

        return self.fc(x)

    def _remap_targets(self, target: torch.Tensor) -> torch.Tensor:
        """Remaps the target to fit with CrossEntropyLoss"""
        mapped = torch.tensor([self.label_map[int(t)] for t in target], device=target.device)
        return mapped.long()

    def training_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        data, target = batch
        data = data.float()
        target = self._remap_targets(target)
        preds = self(data)
        loss = self.criterium(preds.float(), target)
        self.train_acc(preds, target)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_acc", self.train_acc)

        return loss

    def validation_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        data, target = batch
        data = data.float()
        target = self._remap_targets(target)
        preds = self(data)
        loss = self.criterium(preds.float(), target)

        self.log("val_loss", loss)

        return loss

    def test_step(self, batch: torch.utils.data.DataLoader, batch_idx: int):
        data, target = batch
        data = data.float()
        target = self._remap_targets(target)
        preds = self(data)
        loss = self.criterium(preds.float(), target)
        self.test_acc(preds, target)

        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)

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
