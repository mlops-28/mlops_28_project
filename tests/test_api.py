from fastapi.testclient import TestClient
from artsy.api import app
import os
import io
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from artsy import _PATH_CONFIGS
import pandas as pd
from PIL import Image

client = TestClient(app)


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/").json()
        assert response["status_code"] == 200
        assert response["message"] == "Welcome to the Wikiart api created for MLOps 2026"


def test_post_prediction():
    with initialize_config_dir(config_dir=_PATH_CONFIGS, job_name="test", version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    def make_test_image(size=(256, 256), format="PNG", color=(128, 128, 128)):
        img = Image.new("RGB", size, color)
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return buf

    with TestClient(app) as client:
        fake_image = make_test_image(size=(256, 256))

        response = client.post(
            "/predict/",
            files={
                "data": ("fake_image.png", fake_image, "image/png"),
            },
        )

        assert response.status_code == 200
        assert response.json()["message"] == "OK"
        assert response.json()["status-code"] == 200

        prediction = response.json()["prediction"]
        style_path = os.path.join(cfg.data.processed_data_path, "styles.txt")
        styles = pd.read_csv(style_path, sep=",")
        label = int(styles.index[styles["style"] == prediction][0])

        assert label in cfg.data.labels_to_keep
