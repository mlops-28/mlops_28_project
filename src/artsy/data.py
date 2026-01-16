import os
import glob
import torch
import lightning as L

from datasets import load_dataset, Dataset
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm

from omegaconf import DictConfig
from typing import Mapping, Optional


class WikiArtModule(L.LightningDataModule):
    "Loads Wikiart dataset from Huggingface, so it is ready to be used for training and testing with the Pytorch Lightning module"

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.seed = cfg.data.hyperparameters.seed
        self.batch_size = cfg.data.hyperparameters.batch_size
        self.image_size = cfg.data.hyperparameters.image_size
        self.processed_data_path = cfg.data.hyperparameters.processed_data_path
        self.nsamples = cfg.data.hyperparameters.nsamples
        self.labels_to_keep = cfg.data.hyperparameters.labels_to_keep
        self.data_split = cfg.data.hyperparameters.train_val_test

        self.transform = v2.Compose(
            [
                v2.Resize(self.image_size),  # resize shortest side to 256
                v2.CenterCrop(self.image_size),  # center crop to 256x256
                v2.Lambda(lambda x: x.convert("RGB")),
                v2.ToImage(),  # uint8 tensor [C,H,W]
                v2.ToDtype(torch.float16, scale=True),  # Converts to float16 and scales [0, 255] -> [0.0, 1.0]
            ]
        )

    def prepare_data(self) -> None:
        """Function to preproccess data. Data is loaded in and filtered by labels. Afterwards, a subset of each class is selected
        and then processed using the transform defined during initialization. The preprocessed data is then saved batch-wise."""
        if glob.glob(os.path.join(self.processed_data_path, "*.pt")):
            return

        def transform_images(data: Mapping) -> Mapping:
            data["image"] = self.transform(data["image"])
            return data

        def save_data_in_batches(ds: Dataset, nsamples: int = 1000) -> None:
            imgs, labels = [], []
            batch_id = 0

            for ex in tqdm(ds):
                imgs.append(ex["image"])
                labels.append(ex["style"])

                if len(imgs) == nsamples:
                    torch.save(
                        torch.stack(imgs), os.path.join(self.processed_data_path, f"images_batch_{batch_id:04d}.pt")
                    )
                    torch.save(
                        torch.stack(labels), os.path.join(self.processed_data_path, f"labels_batch_{batch_id:04d}.pt")
                    )

                    imgs.clear()
                    labels.clear()
                    batch_id += 1

            if imgs:
                # final partial batch
                torch.save(torch.stack(imgs), f"data/processed/images_batch_{batch_id:04d}.pt")
                torch.save(torch.tensor(labels), f"data/processed/labels_batch_{batch_id:04d}.pt")

        print("Loading dataset")
        self.ds = load_dataset("huggan/wikiart", split="train")

        self.ds = self.ds.filter(lambda x: x["style"] in self.labels_to_keep)

        max_per_class = 6450
        counters = {n: 0 for n in self.labels_to_keep}

        def keep_limited(x: Mapping) -> bool:
            label = x["style"]
            if counters[label] <= max_per_class:
                counters[label] += 1
                return True
            return False

        self.ds = self.ds.filter(keep_limited)

        print("Transforming images")
        self.ds = self.ds.with_transform(transform_images)

        # Save data in batches
        print("Saving data in batches")
        save_data_in_batches(self.ds, nsamples=self.nsamples)

    def setup(self, stage: Optional[str] = None) -> None:
        """Data is loaded from the given directory and split into train, val and test"""
        img_files = sorted(glob.glob(f"{self.processed_data_path}/images_batch_*.pt"))
        label_files = sorted(glob.glob(f"{self.processed_data_path}/labels_batch_*.pt"))

        images = torch.cat([torch.load(f) for f in img_files], dim=0)
        labels = torch.cat([torch.load(f) for f in label_files], dim=0)

        self.dataset = TensorDataset(images, labels)

        self.trainset, self.valset, self.testset = random_split(
            self.dataset, self.data_split, generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        "Initialize Dataloader with training data"
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        "Initialize Dataloader with validation data"
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        "Initialize Dataloader with test data"
        return DataLoader(self.testset, batch_size=self.batch_size)
