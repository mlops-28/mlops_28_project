import time

import argparse
from evidently import Report
from evidently.presets import DataDriftPreset
from hydra import compose, initialize_config_dir
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from artsy import _PATH_CONFIGS
from artsy.data import WikiArtModule


class ArtAPIDataset(Dataset):
    """Initialize Art API dataset."""

    def __init__(self, path_to_csv: str) -> None:
        self.test_images = pd.read_csv(path_to_csv, header=0, index_col=None).reset_index()
        # Test images is now a dataframe with columns img and prediction, where img is a Pytorch tensor (because the data is
        # already preprocessed) and a prediction

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.test_images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item from dataset."""
        img = torch.load(self.test_images.iloc[idx]["img"], weights_only=False)
        # label = int(self.test_images.iloc[idx]["prediction"])
        return img


def extract_features_batch(dataloader):
    features = []

    for batch in dataloader:
        if isinstance(batch, (tuple, list)) and isinstance(batch[0], torch.Tensor):
            images = batch[0]  # shape [B,3,H,W]
        else:
            images = batch  # shape [B,3,H,W]

        images_np = images.numpy().astype(np.float32)

        for img_np in images_np:
            avg_brightness = np.mean(img_np)
            contrast = np.std(img_np) + 1e-6

            # gradient over spatial dimensions only
            grad = np.gradient(img_np, axis=(1, 2))
            sharpness = np.mean([np.mean(np.abs(g)) for g in grad]) + 1e-6

            features.append([avg_brightness, contrast, sharpness])

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    ## Load new data
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_to_csv", default="prediction_database.csv", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-max_workers", default=4, type=int)
    parser.add_argument("-batches_to_check", default=50)
    parser.add_argument("-get_timing", action="store_false")
    args = parser.parse_args()

    # Define dataset
    dataset = ArtAPIDataset(args.path_to_csv)

    # If we want to do see performance over workers
    if args.get_timing:
        mean_time_over_batches = []
        std_time_over_batches = []

        for i in range(args.max_workers):
            # Define dataloader
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=i + 1)

            # lets do some repetitions
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, _ in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)

            res = np.array(res)
            mean_run = np.mean(res)
            std_run = np.std(res)
            mean_time_over_batches.append(mean_run)
            std_time_over_batches.append(std_run)
            print(f"Timing: {mean_run:.3f} +- {std_run:.3f}")

        plt.figure()
        plt.title("Average run time for loading data using a different number of workers")
        plt.xlabel("Number of workers")
        plt.ylabel("Average run time")
        plt.errorbar(list(np.arange(1, args.max_workers + 1)), mean_time_over_batches, yerr=std_time_over_batches)
        plt.savefig("run_time_nworkers.png")
        plt.close()

    # Load old data
    with initialize_config_dir(config_dir=_PATH_CONFIGS, job_name="test", version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    data_module = WikiArtModule(cfg)
    data_module.setup()
    old_dataset = data_module.dataset

    print("Get features of old dataset")
    old_loader = DataLoader(old_dataset, batch_size=args.batch_size, shuffle=False)
    features_old = extract_features_batch(old_loader)

    print("Get features of new dataset")
    new_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    features_new = extract_features_batch(new_loader)

    feature_columns = ["Average Brightness", "Contrast", "Sharpness"]

    breakpoint()

    current_data = pd.DataFrame(features_new, columns=feature_columns)
    reference_data = pd.DataFrame(features_old, columns=feature_columns)

    print("Creating report")
    report = Report(metrics=[DataDriftPreset()])

    result = report.run(reference_data=reference_data, current_data=current_data)

    try:
        report.save_html("data_drifting_report.html")
    except AttributeError:
        result.save_html("data_drifting_report.html")
