import os

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from datasets.floorplans import load_train_data
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

colors = {
    'R3D': 'tab:orange',
    'CubiCasa': 'tab:blue',
    'MURF': 'tab:green',
    'Merged': 'tab:red',
    'CVC-FP': 'tab:brown'
}


def dimensions(dataset, data_dir='data/', imgs_dir='data/annotations/hdd/'):
    paths = get_all_paths(dataset, data_dir)

    minSize = 100000
    minImg = ''
    maxSize = 0
    maxImg = ''
    minDim = (100000, 100000)
    minDimImg = ''
    maxDim = (0, 0)
    maxDimImg = ''
    for path in tqdm(paths, 'Processing'):
        img = cv2.imread(imgs_dir + path.split('\t')[0])

        minSize = min(min(img.shape[0], img.shape[1]), minSize)
        maxSize = max(max(img.shape[0], img.shape[1]), maxSize)

        if img.shape[0] * img.shape[1] < minDim[0] * minDim[1]:
            minDim = img.shape
            minDimImg = path.split('\t')[0].split('/')[1]
        elif img.shape[0] * img.shape[1] > maxDim[0] * maxDim[1]:
            maxDim = img.shape
            maxDimImg = path.split('\t')[0].split('/')[1]

    print(minSize, minImg)
    print(maxSize, maxImg)
    print(minDim, minDimImg)
    print(maxDim, maxDimImg)
    print("Done")


def get_all_paths(dataset, data_dir='data/'):
    train_paths = open(os.path.join(data_dir, dataset + '_train.txt'), 'r').read().splitlines()
    val_paths = open(os.path.join(data_dir, dataset + '_val.txt'), 'r').read().splitlines()
    test_paths = open(os.path.join(data_dir, dataset + '_test.txt'), 'r').read().splitlines()
    paths = train_paths + val_paths + test_paths
    print(f'Dataset: {dataset} Num samples: {len(paths)}')
    print(f"Train_n: {len(train_paths)} Val_n: {len(val_paths)} Test_n: {len(test_paths)}")
    return paths


def get_statistics(dataset, classes, data_dir='data/', imgs_dir='data/annotations/hdd/'):
    paths = get_all_paths(dataset, data_dir)

    # paths = paths[:int(len(paths)/7)]
    # paths = paths[:int(len(paths)/5)]
    xs = []
    ys = []
    weights = [0 for _ in classes]
    frequencies = [[] for _ in classes]
    for path in tqdm(paths, 'Processing'):
        img = cv2.imread(imgs_dir + path.split('\t')[0])
        xs.append(img.shape[0])
        ys.append(img.shape[1])

        mask = cv2.imread(imgs_dir + path.split('\t')[1], cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        for c in range(len(classes)):
            weights[c] += np.count_nonzero(mask == c)
            frequencies[c].append(np.count_nonzero(mask == c) / (mask.shape[0] * mask.shape[1]))

    return xs, ys, weights, frequencies


def bin_dims(data, labels, type, save=True):
    # plt.figure(dpi=400)
    # plt.hist([[w * h for (w, h) in zip(d[0], d[1])] for d in data], bins=20)
    # plt.legend(labels, loc="upper right")
    # plt.yscale('log')
    # plt.show()

    fig, ax = plt.subplots() #plt.figure(dpi=300)
    for i, d in enumerate(data):
        # plt.scatter(d[0], d[1], alpha=0.5, lw=0, c=colors[labels[i]])
        ax.scatter(d[0], d[1], alpha=0.333, lw=0, c=colors[labels[i]], marker='*')
    ax.set_xlabel('Height (pixels)')
    ax.set_ylabel('Width (pixels)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save:
        fig.savefig('bin_dims_' + type, bbox_inches='tight')
    else:
        plt.show()


def plot_areas(data, ticks=None, labels=None, type='', save=True, width=0.2):
    if ticks is not None:
        ticks = [t[0].upper() + t[1:] for t in ticks]
    x_pos = np.arange(len(data[0]), dtype=np.float)
    
    fig, ax = plt.subplots()
    for i, val in enumerate(data):
        offset = i * width
        ax.bar(x_pos + offset, val, width, color=colors[labels[i]])

    ax.set_xticks(x_pos + (np.floor_divide(len(data), 2) * width), ticks)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yscale('log')
    ax.set_ylabel('Number of pixels')
    ax.legend(labels, loc="upper right")
    if save:
        fig.savefig('plot_areas_' + type, bbox_inches='tight')
    else:
        plt.show()


def plot_frequencies(data, ticks=None, labels=None, type='', save=True):
    if ticks is None:
        ticks = []
    for t in range(len(ticks)):
        points = []
        weights = []
        for d in data:
            points.append(d[t])
            weights.append(np.ones_like(d[t]) / len(d[t]))
        
        fig, ax = plt.subplots()
        # plt.hist(points, weights=weights, color=[colors[l] for l in labels], bins=8)
        ax.hist(points, weights=weights, color=[colors[l] for l in labels], bins=12)
        ax.set_xlabel('Ratio of pixels')
        ax.set_ylabel('Normalized frequency')
        ax.legend(labels, loc="upper right")
        if save:
            fig.savefig('plot_frequencies_{0}_{1}'.format(type, ticks[t]), bbox_inches='tight')
        else:
            plt.show()


def map_weights(weights, mapping, ticks):
    weights_map = [0 for _ in ticks]
    for i in range(len(weights)):
        if mapping[i] is not None:
            weights_map[mapping[i]] += weights[i]
    return weights_map


def map_frequencies(weights, mapping, ticks):
    weights_map = [[] for _ in ticks]
    for i in range(len(weights)):
        if mapping[i] is not None:
            weights_map[mapping[i]].append(sum(weights[i])/len(weights[i]))

    return weights_map #[sum(w) / len(w) for w in weights_map]

def inspect_tfrecords(dataset, classes, data_dir='data/'):
    if os.path.exists(os.path.join(data_dir, dataset + '_train.txt')):
        # Print sizes
        _ = get_all_paths(dataset, data_dir)
    train_dataset, validation_dataset, test_dataset = load_train_data(classes, dataset, normalize=False,
                                                                      buffer_size=1, base_dir=data_dir)
    xs, ys = [], []
    weights = [0 for _ in classes]
    frequencies = [[] for _ in classes]
    for data, name in [(train_dataset, 'train'), (validation_dataset, 'val'), (test_dataset, 'test')]:
        for img, mask in tqdm(data, f'Processing {name}'):
            xs.append(img.shape[0])
            ys.append(img.shape[1])
            for c in range(len(classes)):
                weights[c] += np.count_nonzero(mask[..., c])
                frequencies[c].append(np.count_nonzero(mask[..., c]) / (mask.shape[0] * mask.shape[1]))
    return xs, ys, weights, frequencies


if __name__ == '__main__':
    r3d_classes = ['background', 'walls', 'openings']
    cubi_classes = ['background', 'walls', 'railings', 'doors', 'windows', 'stairs']
    multi_classes = ['background', 'walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs']

    multi_r3d_mapping = [0, 1, 1, 1, 2, 2, 2, None]
    multi_cubi_mapping = [0, 1, 1, 2, 3, 3, 4, 5]

    # r3d_xs, r3d_ys, r3d_weights, r3d_frequencies = get_statistics('r3d', r3d_classes,
    #                                                              data_dir='../annotations/r3d/',
    #                                                              imgs_dir='/media/kratochvila/Data/Datasets/')
    # bin_dims([(r3d_xs, r3d_ys)], labels=['R3D'], type='test', save=False)
    # plot_areas([r3d_weights], r3d_classes, ['R3D'], type='test', save=False)
    # exit(0)

    #tfrecords = [('cubicasa5k_augment_multi_plans_augment', multi_classes, 'Merged'),
    #             ('cubicasa5k_augment', multi_classes, 'CubiCasa'),
    #             ('multi_plans_augment', multi_classes, 'MURF'),
    #             ('r3d_augment', r3d_classes, 'R3D')]
    # tfrecords = [('cubicasa5k', multi_classes, 'CubiCasa'), ('r3d', r3d_classes, 'R3D')]
    # '../annotations/'+dataset+'/'
    tfrecords = [('cvc-fp_augment', multi_classes, 'CVC-FP')]
    compare_on_R3D = False
    if tfrecords:
        data = []
        weights = []
        freqs = []
        labels = []
        names = []
        for dataset, classes, label in tfrecords:
            xs, ys, weight, freq = inspect_tfrecords(dataset, classes, data_dir='../data/')
            data.append((xs, ys))
            if classes is multi_classes and compare_on_R3D:
                weights.append(map_weights(weight, multi_r3d_mapping, r3d_classes))
                freqs.append(map_frequencies(freq, multi_r3d_mapping, r3d_classes))
            else:
                weights.append(weight)
                freqs.append(freq)
            labels.append(label)
            names.append(dataset)
        bin_dims(data, labels=labels, type='-'.join(names), save=True)
        plot_areas(weights, ticks=r3d_classes if compare_on_R3D else multi_classes, labels=labels, type='-'.join(names), save=True)
        plot_frequencies(freqs, ticks=r3d_classes if compare_on_R3D else multi_classes, labels=labels, type='-'.join(names), save=True)
        exit(0)

    plot_r3d = False
    plot_cubi = False
    plot_multi = False
    plot_cvc = False
    save_npy = False
    load_npy = False

    r3d_xs, r3d_ys, r3d_weights, r3d_frequencies = get_statistics('r3d', r3d_classes)
    cubi_xs, cubi_ys, cubi_weights, cubi_frequencies = get_statistics('cubicasa5k', multi_classes)
    multi_xs, multi_ys, multi_weights, multi_frequencies = get_statistics('multi_plans', multi_classes)
    cvc_xs, cvc_ys, cvc_weights, cvc_frequencies = get_statistics('CVC-FP', multi_classes,
                                                                  'annotations/CVC-FP',
                                                                  'data_new/CVC-FP')

    if save_npy:
        np.save('r3d_xs', r3d_xs)
        np.save('r3d_ys', r3d_ys)
        np.save('r3d_weights', r3d_weights)
        np.save('r3d_frequencies', r3d_frequencies)
        np.save('cubi_xs', cubi_xs)
        np.save('cubi_ys', cubi_ys)
        np.save('cubi_weights', cubi_weights)
        np.save('cubi_frequencies', cubi_frequencies)
        np.save('multi_xs', multi_xs)
        np.save('multi_ys', multi_ys)
        np.save('multi_weights', multi_weights)
        np.save('multi_frequencies', multi_frequencies)
    if load_npy:
        r3d_xs = np.load('r3d_xs.npy')
        r3d_ys = np.load('r3d_ys.npy')
        r3d_weights = np.load('r3d_weights.npy')
        r3d_frequencies = np.load('r3d_frequencies.npy')
        cubi_xs = np.load('cubi_xs.npy')
        cubi_ys = np.load('cubi_ys.npy')
        cubi_weights = np.load('cubi_weights.npy')
        cubi_frequencies = np.load('cubi_frequencies.npy')
        multi_xs = np.load('multi_xs.npy')
        multi_ys = np.load('multi_ys.npy')
        multi_weights = np.load('multi_weights.npy')
        multi_frequencies = np.load('multi_frequencies.npy')

    if plot_r3d:
        cubi_weights_r3d = map_weights(cubi_weights, multi_r3d_mapping, r3d_classes)
        multi_weights_r3d = map_weights(multi_weights, multi_r3d_mapping, r3d_classes)

        cubi_frequencies_r3d = map_frequencies(cubi_frequencies, multi_r3d_mapping, r3d_classes)
        multi_frequencies_r3d = map_frequencies(multi_frequencies, multi_r3d_mapping, r3d_classes)

        labels = ['R3D', 'CubiCasa', 'MURF']
        bin_dims(data=[(r3d_xs, r3d_ys), (cubi_xs, cubi_ys), (multi_xs, multi_ys)], labels=labels, type='r3d')
        plot_areas([r3d_weights, cubi_weights_r3d, multi_weights_r3d], ticks=r3d_classes, labels=labels, type='r3d')
        plot_frequencies([r3d_frequencies, cubi_frequencies_r3d, multi_frequencies_r3d], ticks=r3d_classes, labels=labels, type='r3d')

    if plot_cubi:
        cubi_weights_cubi = map_weights(cubi_weights, multi_cubi_mapping, cubi_classes)
        multi_weights_cubi = map_weights(multi_weights, multi_cubi_mapping, cubi_classes)

        cubi_frequencies_cubi = map_frequencies(cubi_frequencies, multi_cubi_mapping, cubi_classes)
        multi_frequencies_cubi = map_frequencies(multi_frequencies, multi_cubi_mapping, cubi_classes)

        labels = ['CubiCasa', 'MURF']
        bin_dims(data=[(cubi_xs, cubi_ys), (multi_xs, multi_ys)], labels=labels, type='cubi')
        plot_areas([cubi_weights_cubi, multi_weights_cubi], ticks=cubi_classes, labels=labels, type='cubi')
        plot_frequencies([cubi_frequencies_cubi, multi_frequencies_cubi], ticks=cubi_classes, labels=labels, type='cubi')

    if plot_multi:
        labels = ['MURF']
        bin_dims(data=[(multi_xs, multi_ys)], labels=labels, type='multi')
        plot_areas([multi_weights], ticks=multi_classes, labels=labels, type='multi')
        plot_frequencies([multi_frequencies], ticks=multi_classes, labels=labels, type='multi')

    if plot_cvc:
        labels = ['CVC-FP']
        bin_dims(data=[(cvc_xs, cvc_ys)], labels=labels, type='multi')
        plot_areas([cvc_weights], ticks=multi_classes, labels=labels, type='multi')
        plot_frequencies([cvc_frequencies], ticks=multi_classes, labels=labels, type='multi')
