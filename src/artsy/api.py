from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from http import HTTPStatus
import torch
from datetime import datetime
import os
import pandas as pd
from PIL import Image
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from contextlib import asynccontextmanager

from artsy import _PATH_CONFIGS
from artsy.model import ArtsyClassifier
from artsy.data import WikiArtModule


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# os.makedirs("data/api", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("API starting...")

    global model, datasetup

    print("Loading model")
    with initialize_config_dir(config_dir=_PATH_CONFIGS, job_name="test", version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    datasetup = WikiArtModule(cfg)

    print("Downloading model from WandB")

    model = ArtsyClassifier.load_from_checkpoint(
        checkpoint_path=cfg.eval.model_checkpoint, cfg=cfg, strict=True, map_location=DEVICE, weights_only=False
    )
    model.to(DEVICE)
    model.eval()
    if os.path.exists("/gcs/wikiart-data-api"):
        with open("/gcs/wikiart-data-api/prediction_database.csv", "a") as file:
            file.write("img,prediction\n")
    else:
        with open("data/prediction_database.csv", "w") as file:
            file.write("img,prediction\n")

    yield

    print("Cleaning up")
    del model, datasetup

    print("API shutting down")


app = FastAPI(lifespan=lifespan)


def add_to_database(timestamp: str, img: torch.Tensor, prediction: int) -> None:
    """Simple function to save image path and prediction to database."""

    # Save image (for now just do it locally, the files should be small)
    filename = f"{timestamp}.pt"

    if os.path.exists("/gcs/wikiart-data-api"):
        file_path = "/gcs/wikiart-data-api"
        torch.save(img, os.path.join(file_path, "api", filename))
        with open(os.path.join(file_path, "prediction_database.csv"), "a") as file:
            file.write(f"{timestamp},{file_path},{prediction}\n")
    else:
        torch.save(img, os.path.join("data/api", filename))
        with open("data/prediction_database.csv", "a") as file:
            file.write(f"{timestamp},{file_path},{prediction}\n")


@app.get("/")
def root():
    """Homepage"""
    response = {
        "message": "Welcome to the Wikiart api created for MLOps 2026",
        "status_code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def get_prediction(background_tasks: BackgroundTasks, data: UploadFile = File(...)):
    # Load input image
    input_image = Image.open(data.file)

    # Transform image so that it matches our data
    img = datasetup.transform(input_image)

    # Send image through model
    img = img.to(DEVICE)
    logits = model(img.unsqueeze(0))

    prediction_idx = torch.argmax(logits, dim=1)
    prediction = list(model.label_map.keys())[prediction_idx]
    prediction = int(prediction)

    # Save image and prediction (?)
    timestamp = str(datetime.utcnow().timestamp())
    background_tasks.add_task(add_to_database, timestamp, img, prediction)

    # Get real label
    if os.path.exists("/gcs/wikiart-data-api"):
        style_path = "/gcs/wikiart-data-processed/data/processed/styles.txt"
    else:
        style_path = "data/processed/styles.txt"

    styles = pd.read_csv(style_path, sep=",")
    label = str(styles.at[prediction, "style"])

    # Create response
    response = {
        "prediction": label,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

    return response
