import sys
import os
import cv2
import numpy as np

DATA_DIR = './data/'
NUM_CLASSES = 2
IMAGE_SIZE = 128
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_images(files):
    length = len(files)
    print('length: ' + str(length))
    splitter = int(length * 0.80)
    print('splitter: ' + str(splitter))
    train_files = files[:splitter]
    test_files = files[splitter + 1:]
    print(train_files)
    print(len(train_files))
    print(test_files)
    print(len(test_files))
    sys.exit()

    for datum in data:
        print(datum)
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


def get_images(inputs):
    files = list()

    for i, x in enumerate(inputs):
        for file in os.listdir(x):
            files.append([x + '/' + file, i])

    files = np.array(files)
    files = np.random.permutation(files)

    print(files)

    return read_images(files)


if __name__ == '__main__':
    print(get_images(sys.argv[1:]))
