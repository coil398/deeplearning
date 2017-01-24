import sys
import os
import cv2
import numpy as np

DATA_DIR = './data/'
NUM_CLASSES = 2
IMAGE_SIZE = 118
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_images(data):
    images = list()
    labels = list()

    for datum in data:
        i = cv2.imread(datum[0])
        i = cv2.resize(i, (IMAGE_SIZE, IMAGE_SIZE))
        i = i.flatten().astype(np.float32) / 255.0
        images.append(i)

        tmp = np.zeros(NUM_CLASSES)
        tmp[int(datum[1])] = 1
        labels.append(tmp)

    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels


def main(inputs):
    data = list()

    for i, x in enumerate(inputs):
        files = os.listdir(DATA_DIR + x)
        for f in files:
            data.append([DATA_DIR + inputs[i] + '/' + f, i])

    return read_images(data)


if __name__ == '__main__':
    main(sys.argv[1:])
