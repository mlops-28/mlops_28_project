import torch
import typer

from artsy.data import test_dataloader
from artsy.model import ArtsyClassifier

def evaluate(model_checkpoint: str = "models/model.pth"):
    """Evaluating the trained art classification model"""
    print("Evaluating the trained model")
    
    model = ArtsyClassifier()
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))
    
    test_dataloader = test_dataloader()

    model.eval()

    correct, total = 0, 0

    for images, labels in test_dataloader:
        y_pred = model(images)

        correct += (y_pred.argmax(dim=1) == labels).float().sum().item()

        total += labels.size(0)

        accuracy = correct / total

        print(f"The test accuracy is {accuracy}")

if __name__ == "__main__":
    typer.run(evaluate)