from constants import *
import struct
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistGeneral(Dataset):
    images = list()
    labels = list()

    def __init__(self, image_paths: list[str], label_paths: list[str]) -> None:
        for image_path, label_path in zip(image_paths, label_paths):
            self.__images_from_path(image_path)
            self.__labels_from_path(label_path)

    def __images_from_path(self, path: str) -> None:
        with open(path, "rb") as file:
            magic, num_images, rows, columns = struct.unpack(
                ">IIII", file.read(4 * 4))
            for _ in range(num_images):
                self.images.append(np.frombuffer(
                    file.read(rows * columns), dtype=np.ubyte).reshape(rows, columns))

    def __labels_from_path(self, path: str) -> None:
        with open(path, "rb") as file:
            magic, num_labels = struct.unpack(">II", file.read(2 * 4))
            for _ in range(num_labels):
                self.labels.append(np.frombuffer(
                    file.read(1), dtype=np.ubyte))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
        if torch.is_tensor(index):
            index = index.to_list()

        image = torch.tensor(
            self.images[index], dtype=torch.float32).to(DEVICE)
        label = torch.tensor(
            self.labels[index], dtype=torch.int).flatten().to(DEVICE)
        sample = (image, label)

        return sample


class MnistTrain(MnistGeneral):
    def __init__(self) -> None:
        super().__init__([TRAIN_IMAGES_PATH], [TRAIN_LABELS_PATH])


class MnistTest(MnistGeneral):
    def __init__(self) -> None:
        super().__init__([TEST_IMAGES_PATH], [TEST_LABELS_PATH])


if __name__ == "__main__":
    image_paths = [TRAIN_IMAGES_PATH, TEST_IMAGES_PATH]
    label_paths = [TRAIN_LABELS_PATH, TEST_LABELS_PATH]
    m = MnistGeneral(image_paths, label_paths)
    print(len(m.images))
