import cv2
import struct
from constants import TEST_IMAGES_PATH
import numpy as np


image = np.ndarray

with open(TEST_IMAGES_PATH, "rb") as file:
    magic, num_images, rows, columns = struct.unpack(
        ">IIII", file.read(4 * 4))
    image = np.frombuffer(
        file.read(rows * columns), dtype=np.ubyte).reshape(rows, columns)
    # from matplotlib import pyplot as plt
    # plt.imshow(image)
    # plt.show()

model = cv2.dnn.readNetFromONNX("models/my_model.onnx")
image = cv2.dnn.blobFromImage(image)
model.setInput(image)
print(np.argmax(model.forward()))
