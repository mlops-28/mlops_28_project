from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
import torchmetrics

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from artsy import _PATH_CONFIGS


class ArtsyClassifier(LightningModule):
    """CNN with 3 convolutional layers to classify the artstyle of 128x128 images."""

    def __init__(self, cfg: DictConfig) -> None:
        # def __init__(self, lr: float = 1e-3, drop_p: float = 0.2) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Out: floor((in + 2*padding - kernel_size) / stride) + 1
        self.conv1 = nn.Conv2d(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels1,
            kernel_size=cfg.model.kernel_size1,
            stride=cfg.model.stride1,
        )
        self.conv2 = nn.Conv2d(
            cfg.model.out_channels1, cfg.model.out_channels2, cfg.model.kernel_size2, cfg.model.stride2
        )
        self.conv3 = nn.Conv2d(
            cfg.model.out_channels2, cfg.model.out_channels3, cfg.model.kernel_size3, cfg.model.stride3
        )
        self.fc = nn.LazyLinear(cfg.model.num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.model.drop_p)
        self.max_pool_kernel = cfg.model.max_pool_kernel
        self.max_pool_stride = cfg.model.max_pool_stride
        self.criterium = nn.CrossEntropyLoss()
        self.lr = cfg.model.lr

        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=cfg.model.num_classes)

        self.label_map = cfg.model.label_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))  # Out: floor((128 + 0 - 5) / 2) + 1 = 62
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)  # Out: floor(62 / 2) = 31
        x = self.relu(self.conv2(x))  # Out: floor((31 + 0 - 3) / 1) + 1 = 29
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)  # Out: floor(29 / 2) = 14
        x = self.relu(self.conv3(x))  # Out: floor((14 + 0 - 3) / 1) + 1 = 12
        x = torch.max_pool2d(x, self.max_pool_kernel, self.max_pool_stride)  # Out: floor(12 / 2) = 6
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
        self.val_acc(preds, target)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

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
    with initialize_config_dir(config_dir=_PATH_CONFIGS, version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")
    model = ArtsyClassifier(cfg)
    dummy_input = torch.randn(1, 3, 128, 128)
    output = model(dummy_input)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Output shape: {output.shape}")
