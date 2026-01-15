import pickle
import os
import glob
import torch
from datasets import load_dataset
import lightning as L
from torch.utils.data import random_split, DataLoader
from torch.utils.data import TensorDataset
from torchvision.transforms import v2


class WikiArtModule(L.LightningDataModule):
    def __init__(self, seed: int = 42, batch_size: int = 32, image_size: int = 256, num_workers: int = 4, 
                 processed_data_path: str = None, nsamples: int = 1000):
        super().__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.processed_data_path = processed_data_path
        self.nsamples = nsamples

        self.transform = v2.Compose([
            v2.Resize(256),      # resize shortest side to 256
            v2.CenterCrop(256),  # center crop to 256x256
            v2.Lambda(lambda x: x.convert("RGB")),
            v2.ToImage(),     # uint8 tensor [C,H,W]
            v2.ToDtype(torch.float32, scale=True), # Converts to float32 and scales [0, 255] -> [0.0, 1.0]
        ])
    
    def prepare_data(self):

        if os.path.exists(self.processed_data_path+"/styles.pkl"):
            return

        def transform_images(data):
            data["image"] = self.transform(data["image"])
            return data

        def save_data_in_batches(ds, nsamples = 1000):

            imgs, labels = [], []
            batch_id = 0

            for ex in ds:
                imgs.append(ex["image"])
                labels.append(ex["label"])

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

        self.unique_styles = sorted(set(self.ds.unique("style")))
        self.style_to_id = {s: i for i, s in enumerate(self.unique_styles)}

        self.ds = self.ds.map(lambda x: {"label": self.style_to_id[x["style"]]}, num_proc=1)

        print("Save style-label dictionary")
        with open('data/processed/styles.pkl', "wb") as file:
            pickle.dump(self.style_to_id, file)

        print("Transforming images")
        self.ds = self.ds.with_transform(transform_images)
    
        # Save data in batches
        print("Saving data in batches")
        save_data_in_batches(self.ds, nsamples=self.nsamples)


    def setup(self, stage = None):
        img_files = sorted(glob.glob(f"{self.processed_data_path}/images_batch_*.pt"))
        label_files = sorted(glob.glob(f"{self.processed_data_path}/labels_batch_*.pt"))

        images = torch.cat([torch.load(f) for f in img_files], dim=0)
        labels = torch.cat([torch.load(f) for f in label_files], dim=0)

        self.dataset = TensorDataset(images, labels)

        with open(self.processed_data_path+"/styles.pkl", "rb") as file:
            self.style_to_id = pickle.load(file)

        self.trainset, self.valset, self.testset = random_split(
                self.dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(self.seed)
                )
    
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size)
        

# def preprocess_data():
#     # Load dataset
#     ds = load_dataset("huggan/wikiart", split="train")
#     # data = ds['train']
#     # print(data.features)

#     # Define transform
#     transform = v2.Compose([
#         v2.Resize(256),      # resize shortest side to 256
#         v2.CenterCrop(256),  # center crop to 256x256
#         v2.Lambda(lambda x: x.convert("RGB")),
#         v2.ToImage(),     # uint8 tensor [C,H,W]
#         v2.ToDtype(torch.float32, scale=True), # Converts to float32 and scales [0, 255] -> [0.0, 1.0]
#         #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #ImageNet values
#     ])

#     def transform_images(data):
#         data["image"] = transform(data["image"])
#         return data

#     ds = ds.with_transform(transform_images)

#     # ## Preprocess data
#     # images = []
#     # styles = []

#     # for example in tqdm(data):
#     #     img = transform(example["image"])
#     #     images.append(img)
#     #     styles.append(example["style"])

#     # images = torch.stack(images)

#     # Convert style label to integers
#     # unique_styles = sorted(set(styles))
#     unique_styles = sorted(set(ds.unique("style")))
#     style_to_id = {s: i for i, s in enumerate(unique_styles)}

#     def label_transform(data):
#         data["label"] = style_to_id[data["style"]]
#         return data
    
#     ds = ds.map(label_transform, num_proc=1)

#     # style_ids = torch.tensor([style_to_id[s] for s in styles], dtype=torch.long)

#     ## Split into train and test set
#     # train_idx, test_idx = train_test_split(
#     #     range(len(images)),
#     #     test_size=0.2,
#     #     stratify=style_ids,   # keep style distribution similar
#     #     random_state=42
#     # )
#     ds = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

#     # train_images = images[train_idx]
#     # train_labels = style_ids[train_idx]

#     # test_images = images[test_idx]
#     # test_labels = style_ids[test_idx]

#     train_ds = ds["train"]
#     test_ds = ds["test"]

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#     )

#     test_loader = DataLoader(
#         test_ds,
#         batch_size=32,
#         shuffle=False,
#         num_workers=4,
#     )

#     # torch.save(train_images, "data/processed/train_images.pt")
#     # torch.save(train_labels, "data/processed/train_labels.pt")
#     # torch.save(test_images, "data/processed/test_images.pt")
#     # torch.save(test_labels, "data/processed/test_labels.pt")

#     try:
#         style_file = open('data/processed/styles.pkl', 'wb')
#         pickle.dump(style_to_id, style_file)
#         style_file.close()
#     except:
#         print("Could not save styles")


# def calculate_distribution():

#     train_images = torch.load("data/processed/train_images.pt")
#     train_labels = torch.load("data/processed/train_labels.pt")
#     tst_images = torch.load("data/processed/test_images.pt")
#     test_labels = torch.load("data/processed/test_labels.pt")

#     with open("data/processed/styles.pkl", "rb") as file:
#         style_to_id = pickle.load(file)
#         unique_styles = set(style_to_id.keys())

#     ## Calculate style distribution
#     def style_distribution(labels, names):
#         counts = Counter(labels.tolist())
#         return {names[i]: counts[i] for i in counts}
    
#     train_dist = style_distribution(train_labels, unique_styles)
#     test_dist = style_distribution(test_labels, unique_styles)

#     print("Train style distribution:")
#     for style, count in train_dist.items():
#         print(f"{style}: {count}")

#     print("\nTest style distribution:")
#     for style, count in test_dist.items():
#         print(f"{style}: {count}")

if __name__ == "__main__":
    data = WikiArtModule(processed_data_path = "data/processed")



# from pathlib import Path

# import typer
# from torch.utils.data import Dataset


# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, data_path: Path) -> None:
#         self.data_path = data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""

#     def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""

# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(data_path)
#     dataset.preprocess(output_folder)


# if __name__ == "__main__":
#     typer.run(preprocess)
