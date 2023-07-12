import math
import numpy as np
import os
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import psutil

import pahaw_loader


class ClippingsDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        labels_dict: dict[str, str],
        clippings_dict: dict[str, pahaw_loader.Clipping],
    ):
        self.dataset_type = dataset_type
        self.labels_dict = labels_dict
        self.images_keys = list(self.labels_dict.keys())
        self.transform = transforms.ToTensor()
        self.clippings_dict = clippings_dict

    def __len__(self):
        return len(self.labels_dict)

    def get_clipping(self, index):
        image_key = self.images_keys[index]
        return self.clippings_dict[image_key]

    def __getitem__(self, index):
        image_key = self.images_keys[index]
        image = Image.open(
            os.path.join("generated", self.dataset_type, self.images_keys[index])
        )
        image = np.array(ImageOps.grayscale(image))
        image = self.transform(image)
        label = self.labels_dict[image_key]
        return image, label


class ConvolutionalNetwork(torch.nn.Module):
    def __init__(
        self,
        task_number: int,
        clipping_side_size: int,
        clipping_jump_size: int,
        training_epochs: int,
        train_id_list: list[int],
    ):
        super().__init__()

        self.task_number = task_number
        self.clipping_side_size = clipping_side_size
        self.clipping_jump_size = clipping_jump_size
        self.training_epochs = training_epochs
        self.train_id_list = train_id_list
        self.side_size = (math.floor((clipping_side_size - 2) / 2) - 2) // 2
        self.train_results = TrainResults()

        self.conv1 = torch.nn.Conv2d(1, 6, 3, 1)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1)

        self.fc1 = torch.nn.Linear(self.side_size * self.side_size * 16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, X):
        X = torch.nn.functional.relu(self.conv1(X))
        X = torch.nn.functional.max_pool2d(X, 2, 2)
        X = torch.nn.functional.relu(self.conv2(X))
        X = torch.nn.functional.max_pool2d(X, 2, 2)
        X = X.view(-1, self.side_size * self.side_size * 16)
        X = torch.nn.functional.relu(self.fc1(X))
        X = torch.nn.functional.relu(self.fc2(X))
        X = self.fc3(X)
        return X

    def check(
        self,
        task_number: int,
        clipping_side_size: int,
        clipping_jump_size: int,
        training_epochs: int,
        train_id_list: list[int],
    ):
        return (
            self.task_number == task_number
            and self.clipping_side_size == clipping_side_size
            and self.clipping_jump_size == clipping_jump_size
            and self.training_epochs == training_epochs
            and self.train_id_list == train_id_list
        )


class TrainResults(list):
    def __init__(self):
        super().__init__()

    def append(
        self,
        epoch,
        average_train_loss,
        average_train_accuracy,
        average_test_loss,
        average_test_accuracy,
    ):
        super().append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "train_accuracy": average_train_accuracy,
                "test_loss": average_test_loss,
                "test_accuracy": average_test_accuracy,
            }
        )


def train(
    model: ConvolutionalNetwork,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> ConvolutionalNetwork:
    """Entrena un modelo los epochs especificados."""

    torch.manual_seed(47)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_steps = len(train_loader.dataset) // 10  # type: ignore
    test_steps = len(test_loader.dataset) // 10  # type: ignore

    for e in range(epochs):
        model.train()

        total_train_loss = 0
        total_test_loss = 0
        train_correct = 0
        test_correct = 0

        for x_train, y_train in train_loader:
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.detach().item()

            train_correct += (
                (torch.nn.functional.log_softmax(y_pred, dim=1).argmax(1) == y_train)
                .type(torch.float)
                .sum()
                .item()
            )

        with torch.no_grad():
            model.eval()

            for x_test, y_test in test_loader:
                y_pred = model(x_test)
                loss = criterion(y_pred, y_test)

                total_test_loss += loss.detach().item()

                test_correct += (
                    (torch.nn.functional.log_softmax(y_pred, dim=1).argmax(1) == y_test)
                    .type(torch.float)
                    .sum()
                    .item()
                )

        average_train_loss = total_train_loss / train_steps
        average_test_loss = total_test_loss / test_steps

        train_accuracy = train_correct / len(train_loader.dataset)  # type: ignore
        test_accuracy = test_correct / len(test_loader.dataset)  # type: ignore

        model.train_results.append(
            e, average_train_loss, train_accuracy, average_test_loss, test_accuracy
        )

        if (e + 1) % 5 == 0:
            print(
                f"Epoch: {e + 1}/{epochs} | Train loss: {average_train_loss:.4f}, Train"
                + f" accuracy: {(train_accuracy * 100):.2f} | Test loss: "
                + f"{average_test_loss:.4f}, Test accuracy: {(test_accuracy * 100):.2f}"
            )

            print("Memoria usada (GB): ", psutil.virtual_memory()[3] / 1000000000)

    return model


def predict(
    model: ConvolutionalNetwork, temp_dataset: ClippingsDataset, clipping_side_size: int
):
    model.eval()
    i = 0
    while i < len(temp_dataset):
        with torch.no_grad():
            prediction = model(
                temp_dataset[i][0].view(1, 1, clipping_side_size, clipping_side_size)
            ).argmax()
            temp_dataset.get_clipping(i).pd_predicted = int(prediction)
            i += 1
