import io

from fastapi.testclient import TestClient
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
import pandas as pd
from PIL import Image

from artsy import _PATH_CONFIGS
from artsy.api import app

client = TestClient(app)


def test_read_root() -> None:
    """Test response from homepage."""
    with TestClient(app) as client:
        response = client.get("/").json()
        assert response["status_code"] == 200
        assert response["message"] == "Welcome to the Wikiart api created for MLOps 2026"


def test_post_prediction() -> None:
    """Test model prediction in API."""
    with initialize_config_dir(config_dir=_PATH_CONFIGS, job_name="test", version_base=None):
        cfg: DictConfig = compose(config_name="default_config.yaml")

    def make_test_image(
        size: tuple[int, int] = (256, 256), format: str = "PNG", color: tuple[int, int, int] = (128, 128, 128)
    ) -> io.BytesIO:
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
        style_path = "data/processed/styles.txt"
        styles = pd.read_csv(style_path, sep=",")
        label = int(styles.index[styles["style"] == prediction][0])

        assert label in cfg.data.labels_to_keep
