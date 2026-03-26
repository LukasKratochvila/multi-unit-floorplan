import random
from multiprocessing.pool import Pool

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

RESIZE_MODE = 'resize'  # pad
MIN_SIZE = 512
MAX_SIZE = 832
STRIDE = 64

# img_size = 1000
# img_size = 1500
img_size = 2000

def process(i):
    sizes = list(range(MIN_SIZE, MAX_SIZE + STRIDE, STRIDE))
    h, w = img_size, img_size
    prediction_count = np.zeros((h, w))
    size = random.choice(sizes)
    hsize = int(size / 2)

    samples = []

    # First stage: random samples
    points = zip(np.random.randint(0, h, size=1),
                 np.random.randint(0, w, size=1))
    for p in points:
        y = p[0]
        x = p[1]
        y1 = max(0, y - hsize + min(0, h - (y + hsize)))
        y2 = min(h, y + hsize + max(0, -(y - hsize)))
        x1 = max(0, x - hsize + min(0, w - (x + hsize)))
        x2 = min(w, x + hsize + max(0, -(x - hsize)))
        prediction_count[y1:y2, x1:x2] += 1
        samples.append([y1, y2, x1, x2])

    # Second stage: n passes over matrices
    for c in range(1):
        uncovered = np.argwhere(prediction_count == c)
        while uncovered.any():
            points = uncovered[np.random.choice(len(uncovered), size=min(len(uncovered), 1), replace=False)]
            size = random.choice(sizes)
            hsize = int(size / 2)
            for p in points:
                y = p[0]
                x = p[1]
                y1 = max(0, y - hsize + min(0, h - (y + hsize)))
                y2 = min(h, y + hsize + max(0, -(y - hsize)))
                x1 = max(0, x - hsize + min(0, w - (x + hsize)))
                x2 = min(w, x + hsize + max(0, -(x - hsize)))
                prediction_count[y1:y2, x1:x2] += 1
                samples.append([y1, y2, x1, x2])

            uncovered = np.argwhere(prediction_count == c)
    return prediction_count, len(samples)


def sampling():
    img = np.zeros([img_size, img_size])
    h, w = img.shape
    total_count = np.zeros((h, w))

    reload = False
    if reload:
        pool = Pool(processes=22)
        r = pool.map(process, list(range(5000)))
        pool.close()
        pool.join()

        counts = []
        for d in r:
            total_count += d[0]
            counts.append(d[1])

        np.save('total_count_' + str(img_size), total_count)
        np.save('counts_' + str(img_size), counts)

    total_count = np.load('total_count_' + str(img_size) + '.npy')
    counts = np.load('counts_' + str(img_size) + '.npy')

    counts = counts[:5000]
    print(len(counts))

    # plt.imshow(total_count)
    # plt.show()

    print(min(counts))
    print(max(counts))
    plt.hist(counts, bins=18)
    plt.xlabel('#tiles')
    plt.xticks(list(range(10, 30, 2)))
    plt.ylabel('Frequency')
    plt.savefig('test_samples_' + str(img_size), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    sampling()
