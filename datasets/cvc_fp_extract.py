import os
import argparse
import shutil
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from datasets.create_data_mask import create_overlay


def draw(points):
    coord = []
    for point in points.split(' ')[:-1]:
        if point == '':
            continue
        x, y = point.split(',')
        x = float(x)
        y = float(y)
        coord.append([x, y])

    coord.append(coord[0])
    return coord


def create_folder_structure(root_path, overlay):
    gt_subfolder = 'ImagesGT'
    for i in tqdm(os.listdir(os.path.join(root_path, gt_subfolder)), 'Extracting mask files'):
        if '_gt_' in i:
            sub_dir = os.path.join(root_path, i.split('_gt_')[0])
            # Solve input image name and ending
            if os.path.exists(os.path.join(root_path, gt_subfolder, i.split('_gt_')[0] + '.png')):
                source_img = i.split('_gt_')[0] + '.png'
            else:
                source_img = i.split('_gt_')[0] + '.jpg'
            # Create subFolder with input and gt
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            if not os.path.exists(os.path.join(sub_dir, 'input.png')):
                shutil.copy2(os.path.join(root_path, gt_subfolder, source_img), os.path.join(sub_dir, 'input.png'))
            if not os.path.exists(os.path.join(sub_dir, i)):
                shutil.copy2(os.path.join(root_path, gt_subfolder, i), os.path.join(sub_dir, i))
            if not os.path.exists(os.path.join(sub_dir, 'mask.png')):
                # Create mask
                parse_cvc_fp_label(sub_dir, i, 'input.png', overlay)


def parse_cvc_fp_label(path, mask_name, img_name, overlay):
    xml = ET.parse(os.path.join(path, mask_name))
    root = xml.getroot()

    classes = ['Wall', 'Glass_wall', 'Railing', 'Door', 'Sliding_door', 'Window',
               'Stairs_all', 'Room', 'Parking']  #, 'Separation'] # shouldnt be in mask
    elements = {}
    for c in classes:
        elements[c] = []

    for child in root:
        if 'class' in child.attrib and child.attrib['class'] in classes:
            elements[child.attrib['class']].append(draw(child.attrib['points']))

    h, w, _ = cv2.imread(os.path.join(path, img_name)).shape

    for c in classes:
        bg = Image.new('L', size=(w, h))
        image_draw = ImageDraw.Draw(bg)
        for e in elements[c]:
            if len(e) == 2:  # Circle
                image_draw.ellipse(e, fill='white', outline='white', width=0)
            else:  # Polygon
                image_draw.polygon([(xy[0], xy[1]) for xy in e], fill='white', outline='white', width=0)
        bg.save(os.path.join(path, c.lower() + '.png'))

    mask = np.zeros((h, w, 1), np.uint8)
    for c in range(len(classes)):
        c_ind = (cv2.imread(os.path.join(path, classes[c].lower() + '.png'), 0)).astype(np.uint8)
        mask[c_ind == 255] = c + 1  # border
        if classes[c] == 'Room':  # simplify doors and windows
            mask[c_ind == 255] = 0
        elif classes[c] == 'Parking':  # door class
            mask[c_ind == 255] = 4

    cv2.imwrite(os.path.join(path, 'mask.png'), mask)

    if overlay:
        create_overlay('', path, img_name, 'mask.png', alpha=0.8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract each floorplan from CVC-FP dataset and create folder '
                                                 'structure with input and mask image')
    parser.add_argument('--root_path', help='Path to the dataset', default='../../../Datasets/CVC-FP')
    parser.add_argument('--overlay', action='store_true', help='Create overlay to show annotation')

    args = parser.parse_args()

    create_folder_structure(args.root_path, args.overlay)
