from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
import torch
import os
from PIL import Image
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from contextlib import asynccontextmanager

from artsy import _PROJECT_ROOT, _PATH_CONFIGS
from artsy.model import ArtsyClassifier
from artsy.data import WikiArtModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("API starting...")

    global model, datasetup

    print("Loading model")
    with initialize_config_dir(config_dir=_PATH_CONFIGS, job_name="test", version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    datasetup = WikiArtModule(cfg)
    model_checkpoint = os.path.join(_PROJECT_ROOT, cfg.eval.model_checkpoint)
    model = ArtsyClassifier.load_from_checkpoint(checkpoint_path=model_checkpoint, strict=True, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    yield

    print("Cleaning up")
    del model, datasetup

    print("API shutting down")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    """Homepage"""
    response = {
        "message": "Welcome to the Wikiart api created for MLOps 2026",
        "status_code": HTTPStatus.OK,
    }
    return response


@app.post("/predict/")
async def get_prediction(data: UploadFile = File(...)):
    # Load input image
    input_image = Image.open(data.file)

    # Transform image so that it matches our data
    img = datasetup.transform(input_image)

    # Send image through model
    logits = model(img.unsqueeze(0))

    prediction_idx = torch.argmax(logits, dim=1)
    prediction = list(model.label_map.keys())[prediction_idx]

    # Save image and prediction (?)
    #

    # Create response
    response = {
        "prediction": int(prediction),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }

    return response
