import os
import time

from scipy.spatial.distance import cdist
from scipy.ndimage.morphology import distance_transform_edt
import tensorflow as tf
from utils import ind2rgb, rgb2ind
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def distmat(shape, points, method='distance_transform_edt'):
    """
    Function for computing squared Euclidian distance matrix between indices and positions in matrix
    source: https://stackoverflow.com/questions/61628380/calculate-distance-from-all-points-in-numpy-array-to-a-single-point-on-the-basis

    :param shape: Shape of output matrix
    :param points: Points for distance computing
    :param method: Method for computing output, available 'distance_transform_edt' (default), 'hardcoded', 'norm' or 'original'

    :return: Matrix with distances between indices and positions in output matrix
    """
    if method == 'distance_transform_edt':
        # takes 406ms per heatmap
        indices = np.atleast_2d(points)
        mask = np.ones(shape, dtype=bool)
        mask[indices[:, 1], indices[:, 0]] = False
        edt_matrix = distance_transform_edt(mask)
        return edt_matrix ** 2
    elif method == 'hardcoded':
        # takes 50s per heatmap
        indices = np.atleast_2d(points)
        i, j = np.indices(shape, sparse=True)
        return (((i - indices[:, 1])[..., None]) ** 2 + (j - indices[:, 0, None]) ** 2).min(1)
    elif method == 'norm':
        # takes 12s
        heatmap = np.zeros((len(points),) + shape, np.double)
        indices = np.transpose(np.indices(shape), (0, 2, 1))
        for i, point in enumerate(points):
            point = np.array(point, dtype=np.int)
            subtract = indices - point[:, None, None]
            heatmap[i, ...] = np.linalg.norm(subtract, axis=0)
        return np.min(heatmap, axis=0) ** 2
    elif method == 'original':
        # Original - takes 11 min 8 s per heatmap
        heatmap = np.zeros(shape, np.double)
        for y, x in np.ndindex(shape):
            heatmap[y, x] = np.min(cdist(np.array([(x, y)]), points, 'sqeuclidean'))
        return heatmap
    else:
        ValueError(f'Method: {method} is not implemented.')


def write(file, img):
    cv2.imwrite(file, img)


def create_heatmap_from_paths(dataset, data_dir, sub_dirs, show=False):
    print(f'Processing heatmaps for dataset: {dataset} with {len(sub_dirs)} files.')
    config = {'dataset': dataset, 'path': os.path.join(data_dir, dataset)}
    if 'cubicasa5k' in dataset:
        config['classes'] = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        config['classes_indices'] = [3, 5, 6]
    elif 'r3d' in dataset:
        config['classes'] = ['walls', 'openings']
        config['classes_indices'] = [1]
    elif 'multi_plans' in dataset:
        config['classes'] = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        config['classes_indices'] = [3, 4, 5, 6]
    else:
        print(f'Dataset {dataset} is not Implemented')
        return

    for sub_dir in tqdm(sub_dirs):
        process(config, show, sub_dir)


def create_heatmap_from_config(config, show=False):
    config['path'] = os.path.join(config['data_dir'], config['dataset'])
    sub_dirs = os.listdir(config['path'])
    print('Processing config:')
    print(config)
    print('with #: ', len(sub_dirs), ' files.')

    # done_lines = open('hm_cubi.log').readlines()[2:]
    # done = []
    # for l in done_lines:
    #     if l[0] == 'D':
    #         done.append(l.split('  ')[1].split(']')[0] + ']')
    #
    # files = [f for f in files if f not in done]
    # print(len(done))
    # print(len(files))

    # pool = Pool(processes=55)
    pool = Pool(processes=22)
    pool.map(partial(process, config, show), sub_dirs)
    pool.close()
    pool.join()

    # for f in ['31878855.jpg_True_True_[]']:
    #     process(config, False, f)

    # step = 200
    # i = 0
    # start = i*step
    # end = start+step
    # files = files[start:end]
    # for f in files:
    #     process(config, False, f)


def generate_heatmap(c, show, mask):
    img = (mask == c + 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    if show:
        plt.figure(dpi=200)
        plt.imshow(mask)
        # plt.scatter(*zip(*centroids), color='b', s=.5)

    contours_map = np.zeros(img.shape, np.float32)

    endpoints = []
    for stat in stats[1:]:

        x = stat[0] - 1
        y = stat[1] - 1
        w, h = stat[2] + 1, stat[3] + 1
        if show:
            plt.gca().add_patch(Rectangle((x, y), w, h, facecolor='g', alpha=0.6))

        bb = img[y: y + h, x: x + w]
        contours, hierarchy = cv2.findContours(bb, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE, offset=[x, y])

        cv2.drawContours(contours_map, contours, -1, (255, 255, 255), -1)

        xy = []
        for contour in contours:
            for coord in contour:
                p = (coord[0][0], coord[0][1])
                if img[p[1], p[0]] == 1:
                    xy.append(p)

        xy = sorted(xy, key=lambda p: sum(p))
        endpoints += xy
    if show:
        plt.axis('off')
        plt.savefig('heatmap_step1', bbox_inches='tight', pad_inches=0)

        plt.scatter(*zip(*endpoints), color='r', s=.1)
        plt.savefig('heatmap_step2', bbox_inches='tight', pad_inches=0)
        plt.show()

    heatmap = None
    endpoints = endpoints[:-1]
    if len(np.array(endpoints).shape) == 2:
        heatmap = distmat(img.shape, endpoints)
        # Because numpy saving otherwise save fail
        if len(endpoints) in img.shape:
            endpoints = endpoints[:-1]

    return heatmap, endpoints, contours_map


def process(config, show, folder):
    tic = time.time()
    path = config['path']
    input = cv2.imread(os.path.join(path, folder, 'input.png'))
    mask = cv2.imread(os.path.join(path, folder, 'mask.png'), 0)

    if show:
        plt.figure(dpi=200)
        plt.imshow(ind2rgb(mask))
        plt.axis('off')
        plt.savefig('heatmap_label', bbox_inches='tight', pad_inches=0)

    for c in config['classes_indices']:
        heatmap_path = os.path.join(path, folder, 'heatmap_' + config['classes'][c] + '.npy')
        # if True or not os.path.isfile(heatmap_path):
        if not os.path.isfile(heatmap_path):
            data = generate_heatmap(c, show, mask)
            heatmap, endpoints, contours_map = data
            if len(np.array(endpoints).shape) != 2:
                data = [[], endpoints, []]
            np.save(heatmap_path, data)
        else:
            data = np.load(heatmap_path, allow_pickle=True)

        create_heatmap = True
        if create_heatmap:
            img = (mask == c + 1).astype(np.uint8)
            heatmap, endpoints, contours_map = data

            avg_heatmap = np.zeros(img.shape, np.float32)
            if len(np.array(endpoints).shape) != 2:
                print('EMPTY: {0} {1}'.format(folder, config['classes'][c]))
            else:
                sigmas = [2, 10]
                for sigma in sigmas:
                    sigma = sigma ** 2
                    heatmap_beta = np.exp(-heatmap / sigma)

                    heatmap_beta += contours_map / 255.0
                    heatmap_beta[heatmap_beta < 1e-7] = 0.0
                    heatmap_beta[heatmap_beta > 1] = 1.0

                    avg_heatmap += heatmap_beta
                    if show:
                        plt.figure(dpi=200)
                        plt.imshow(heatmap_beta)
                        plt.show()
                        print(np.min(heatmap_beta), np.max(heatmap_beta))

                        plt.figure(dpi=200)
                        plt.imshow(input)
                        plt.imshow(heatmap_beta, alpha=0.6)
                        plt.axis('off')
                        plt.savefig('heatmap_beta' + str(sigma), bbox_inches='tight', pad_inches=0)
                        plt.show()

                avg_heatmap /= len(sigmas)

            write(os.path.join(path, folder, 'heatmap_' + config['classes'][c] + '.png'), avg_heatmap * 255)

            if show:
                # plt.hist(heatmap.flatten())
                # plt.show()

                plt.figure(dpi=200)
                plt.imshow(mask)
                plt.axis('off')
                plt.show()

                print(np.min(heatmap), np.max(heatmap))
                print(np.min(avg_heatmap), np.max(avg_heatmap))

                plt.figure(dpi=200)
                # plt.imshow(input)
                # plt.imshow(avg_heatmap, alpha=0.6)
                plt.imshow(avg_heatmap)
                plt.axis('off')
                plt.savefig('avg_heatmap_betas', bbox_inches='tight', pad_inches=0)
                plt.show()

    toc = time.time()
    print('Done: ', folder, (toc - tic), 's')


def single(config):
    path = 'annotations/hdd/' + config['dataset'] + '/'
    files = os.listdir(path)
    process(config, False, files[0])


def plot_exp():
    x = np.linspace(0, 500, 2000)
    y1 = (np.exp(-x / 2 ** 2) + np.exp(-x / 10 ** 2)) / 2
    y1[y1 < 0.5] = np.nan
    y2 = (np.exp(-x / 2 ** 2) + np.exp(-x / 10 ** 2)) / 2
    y2[y2 >= 0.5] = np.nan
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, [0.5 for _ in x], linestyle='dashed')

    plt.xlabel('Euclidian distance (pixels)')
    plt.ylabel('Heatmap value')
    plt.legend([r'$\beta = 2$', r'$\beta = 10$'], loc='upper right')
    plt.savefig('heatmap_exp.png', pad_inches=0)
    plt.show()


def plot_scales():
    heatmap = cv2.imread('avg_heatmap_betas.png', 0)
    scales = [4, 8, 16]
    for s in scales:
        plt.figure(dpi=200)
        dims = [heatmap.shape[1] / s, heatmap.shape[0] / s]
        dims = [int(x) for x in dims]
        plt.imshow(cv2.resize(heatmap, dims, interpolation=cv2.INTER_AREA))
        plt.axis('off')
        plt.savefig('heatmap_scale{0}'.format(s) + '.png', bbox_inches='tight', pad_inches=0)
        plt.show()


def eval():
    prediction = cv2.cvtColor(cv2.imread('heatmap_label.png'), cv2.COLOR_BGR2RGB)
    prediction = np.expand_dims(rgb2ind(prediction), axis=0)
    y_pred = tf.one_hot(prediction, 3)[:, :, :, 2]

    y_pred = np.random.random(y_pred.shape)

    y_true = np.expand_dims(cv2.imread('avg_heatmap_betas.png', 0) / 255.0, axis=0)

    diff = np.square(y_pred - y_true)
    diff = np.linalg.norm(y_pred - y_true)
    print(diff)
    # loss = np.sum(diff)
    # print(1 - 1/loss)
    # plt.imshow(diff.squeeze())
    # plt.show()


if __name__ == '__main__':
    # eval()
    # exit(0)

    # plot_exp()
    # # plot_scales()
    # exit(0)

    configs = [
        # {
        #     'dataset': 'multi_plans_augment',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'classes_indices': [3, 4, 5, 6],  # TODO maybe add stairs
        # },
        # {
        #     'dataset': 'cubicasa5k_augment',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'classes_indices': [3, 5, 6],
        # },
        {
            'dataset': 'r3d_augment',
            'classes': ['walls', 'openings'],
            'classes_indices': [1],
        },
    ]

    # single(configs[0])
    # exit(0)

    for config in configs:
        tic = time.time()
        create_heatmap_from_config(config)
        toc = time.time()
        print('total evaluation time = {} minutes'.format((toc - tic) / 60))
