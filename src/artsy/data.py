import pickle
import os
import glob
import torch
from datasets import load_dataset
import lightning as L
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.transforms import v2
from tqdm import tqdm
from collections import defaultdict

class WikiArtModule(L.LightningDataModule):
    "Loads Wikiart dataset from Huggingface, so it is ready to be used for training and testing with the Pytorch Lightning module"
    def __init__(self, seed: int = 42, batch_size: int = 32, image_size: int = 128, num_workers: int = 4, 
                 processed_data_path: str = None, nsamples: int = 1000, labels_to_keep: list = [12,21,23,9,20],
                 train_val_test: list = [0.8, 0.1, 0.1]):
        super().__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.processed_data_path = processed_data_path
        self.nsamples = nsamples
        self.labels_to_keep = labels_to_keep
        self.data_split = train_val_test

        self.transform = v2.Compose([
            v2.Resize(self.image_size),      # resize shortest side to 256
            v2.CenterCrop(self.image_size),  # center crop to 256x256
            v2.Lambda(lambda x: x.convert("RGB")),
            v2.ToImage(),     # uint8 tensor [C,H,W]
            v2.ToDtype(torch.float16, scale=True), # Converts to float16 and scales [0, 255] -> [0.0, 1.0]
        ])
    
    def prepare_data(self):
        """ Function to preproccess data. Data is loaded in and filtered by labels. Afterwards, a subset of each class is selected
        and then processed using the transform defined during initialization. The preprocessed data is then saved batch-wise. """
        if glob.glob(os.path.join(self.processed_data_path, "*.pt")):
            return

        def transform_images(data):
            data["image"] = self.transform(data["image"])
            return data

        def save_data_in_batches(ds, nsamples = 1000):

            imgs, labels = [], []
            batch_id = 0

            for ex in tqdm(ds):
                imgs.append(ex["image"])
                labels.append(ex["style"])

                if len(imgs) == nsamples:
                    torch.save(torch.stack(imgs), f"data/processed/images_batch_{batch_id:04d}.pt")
                    torch.save(torch.tensor(labels), f"data/processed/labels_batch_{batch_id:04d}.pt")

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

        def keep_limited(x):
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


    def setup(self, stage = None):
        """ Data is loaded from the given directory and split into train, val and test """
        img_files = sorted(glob.glob(f"{self.processed_data_path}/images_batch_*.pt"))
        label_files = sorted(glob.glob(f"{self.processed_data_path}/labels_batch_*.pt"))

        images = torch.cat([torch.load(f) for f in img_files], dim=0)
        labels = torch.cat([torch.load(f) for f in label_files], dim=0)

        self.dataset = TensorDataset(images, labels)

        self.trainset, self.valset, self.testset = random_split(
                self.dataset, self.data_split, generator=torch.Generator().manual_seed(self.seed)
                )
    
    def train_dataloader(self):
        "Initialize Dataloader with training data"
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def val_dataloader(self):
        "Initialize Dataloader with validation data"
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self):
        "Initialize Dataloader with test data"
        return DataLoader(self.testset, batch_size=self.batch_size)
        

if __name__ == "__main__":
    print("Start-up")
    data = WikiArtModule(processed_data_path = "data/processed")
    data.prepare_data()
    print("Shut-down")
