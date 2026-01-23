import io

from locust import HttpUser, between, task
import numpy as np
from PIL import Image


def make_test_image(size: int = 256) -> io.BytesIO:
    """Generate a test image."""
    data = np.random.rand(size, size, 3)
    data_uint8 = (data * 255).astype(np.uint8)
    img = Image.fromarray(data_uint8, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task(3)
    def get_item(self) -> None:
        """A task that simulates a user posting an image to the api."""
        fake_image = make_test_image(size=256)

        self.client.post(
            "/predict/",
            files={
                "data": ("fake_image.png", fake_image, "image/png"),
            },
        )
