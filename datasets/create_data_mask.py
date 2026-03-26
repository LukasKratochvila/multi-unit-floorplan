import os
import numpy as np
from PIL import Image
from utils import ind2rgb


def create_mask(sub_dirs, path='annotations/hdd/r3d/', classes=None):

    if classes is None:
        classes = ['wall', 'close']  # r2d dataset

    for sub_dir in sub_dirs:
        file_base = os.path.join(path, sub_dir)
        openings = Image.open(os.path.join(file_base, classes[0] + '.png'))

        mask = np.zeros(openings.size, np.uint8)
        for c in range(len(classes)):
            img = Image.open(os.path.join(file_base, classes[c] + '.png')).convert('L')
            mask[np.array(img) == 255] = (c + 1)
        mask = Image.fromarray(mask)
        mask.save(os.path.join(file_base, 'mask.png'))


def create_overlay(sub_dir, data_dir='annotations/hdd/r3d/', img_name='input.jpg', mask_name='mask.png', save=True,
                   alpha=0.6, color_map='floorplan_map'):
    path = os.path.join(data_dir, sub_dir)

    image = Image.open(os.path.join(path, img_name)).convert('RGB')
    mask = Image.open(os.path.join(path, mask_name))

    mask = Image.fromarray(ind2rgb(np.array(mask), color_map=color_map)).resize(image.size)

    overlay = Image.blend(image, mask, alpha)
    if save:
        overlay.save(os.path.join(path, '_'.join([img_name, mask_name, 'overlay.png'])))
    else:
        overlay.show()
