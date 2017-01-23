import sys
import os
import cv2
import numpy as np

DATA_DIR = './data/'
NUM_CLASSES = 2
IMG_SIZE = 118


def read_images(data):
    image = list()
    label = list()

    for datum in data:
        i = cv2.imread(datum[0])
        i = cv2.resize(i, (IMG_SIZE, IMG_SIZE))
        i = i.flatten().astype(np.float32) / 255.0
        image.append(i)

        tmp = np.zeros(NUM_CLASSES)
        tmp[datum[1]] = 1
        label.append(tmp)

        print(i, tmp)

    return image, label


def main(inputs):
    data = list()

    for i, x in enumerate(inputs):
        files = os.listdir(DATA_DIR + x)
        for f in files:
            data.append([DATA_DIR + inputs[i] + '/' + f, i])

    return read_images(data)


if __name__ == '__main__':
    main(sys.argv[1:])
