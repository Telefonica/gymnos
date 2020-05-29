#
#
#   Covid19ChestXrayCnn
#
#

import os
import numpy as np

from .model import Model
from ..utils import lazy_imports, DataLoader

from tqdm import tqdm
from collections import defaultdict

torch = lazy_imports.torch
torchvision = lazy_imports.torchvision


class ClassifierCNN(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class Covid19ChestXrayCnn(Model):
    """
    Task: **Classification**

    Convolutional neuronal network developed to solve COVID-19 chest X-ray image classification
    task (:class:`gymnos.datasets.covid19_chest_xray.Covid19ChestXray`)

    Can I use?
        - Generators: ✔️
        - Probability predictions: ✔️
        - Distributed datasets: ❌

    Parameters
    ----------
    num_classes: int, optional
        Optional number of classes to classify images into. This is useful if
        you want to train this model with another dataset.
    class_weights: list of floats
        Class weights to put into loss function. Useful for imbalanced datasets
    learning_rate: float
        Learning rate for optimizer
    device: str
        Device to train model, "cuda" or "cpu"

    Warnings
    ----------------
    This model requires channels first images.

    Examples
    --------
    .. code-block:: py

        Covid19ChestXrayCnn(
            classes=3,
            class_weights=[0.2. 2.34, 0.5]
        )
    """

    def __init__(self, num_classes=3, class_weights=None, learning_rate=1e-3, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if class_weights is not None:
            class_weights = torch.tensor(class_weights)

        self.device = device

        self.model = ClassifierCNN(num_classes).to(device)

        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y, epochs=5, batch_size=32):
        """
        Fit model to data

        Parameters
        -----------
        epochs: int
            Number of epochs to train
        batch_size: int
            Batch size.

        Returns
        ---------
        dict of str
            Dictionnary with metrics ("loss" and "acc")
        """
        return self.fit_generator(DataLoader(list(zip(X, y)), batch_size=batch_size), epochs=epochs)

    def fit_generator(self, generator, epochs=5):
        """
        Fit model to generator

        Parameters
        -----------
        epochs: int
            Number of epochs to train

        Returns
        ---------
        dict of str
            Dictionnary with metrics ("loss" and "acc")
        """
        self.model.train()

        metrics = defaultdict(list)

        for _ in tqdm(range(epochs), leave=True):
            num_samples = 0
            running_loss = 0.0
            running_correct_labels = 0

            for features, targets in tqdm(generator):
                features = torch.tensor(features, dtype=torch.float32, device=self.device)
                targets = torch.tensor(targets, dtype=torch.long, device=self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                num_samples += len(features)

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()
                running_correct_labels += torch.sum(preds == targets).item()

            metrics["loss"].append(running_loss / len(generator))
            metrics["acc"].append(running_correct_labels / float(num_samples))

        return metrics

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        self.model.eval()

        features = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(self.model(features), dim=1)

        return outputs.cpu().numpy()

    def evaluate(self, X, y, batch_size=32):
        return self.evaluate_generator(DataLoader(list(zip(X, y)), batch_size=batch_size))

    def evaluate_generator(self, generator):
        self.model.eval()

        num_samples = 0
        running_loss = 0.0
        running_correct_labels = 0

        for features, targets in tqdm(generator):
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
            targets = torch.tensor(targets, dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(features)

            loss = self.criterion(outputs, targets)

            _, preds = torch.max(outputs, 1)

            num_samples += len(features)
            running_loss += loss.item()
            running_correct_labels += torch.sum(preds == targets).item()

        return {
            "loss": running_loss / len(generator),
            "acc": running_correct_labels / float(num_samples)
        }

    def save(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pth"))

    def restore(self, save_dir):
        self.model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location=self.device))
