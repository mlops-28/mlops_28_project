import io
from locust import HttpUser, between, task
from PIL import Image


def make_test_image(size=(256, 256), format="PNG", color=(128, 128, 128)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=format)
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
        fake_image = make_test_image(size=(256, 256))

        self.client.post(
            "/predict/",
            files={
                "data": ("fake_image.png", fake_image, "image/png"),
            },
        )
