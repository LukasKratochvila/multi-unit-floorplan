import os
import cv2
import argparse
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from MLStructFP.db._c import GeomPoint2D
from MLStructFP.db import DbLoader

from datasets.create_data_mask import create_overlay
#matplotlib.use('Qt5Agg')

def create_folders_and_masks(root_path, overlay, recreate):
    db = DbLoader(os.path.join(root_path, 'fp.json'))
    data_path = os.path.join(root_path, 'MLSTRUCT-FP_v1')
    keys = list(db.floor.keys())
    pbar = tqdm(keys, "Processing")  # [:1]
    for ID in pbar:
        pbar.set_postfix_str(f'ID: {ID}')
        subDir = os.path.join(data_path, str(ID))
        if not os.path.exists(subDir):
            os.makedirs(subDir)

        # Open floorPlan
        floor = db.floor[ID]
        layout = ImageOps.invert(Image.open(floor.image_path).convert('L'))
        w, h = layout.size

        # Save input to framework
        if not os.path.exists(os.path.join(subDir, 'input.png')) or recreate:
            layout.save(os.path.join(subDir, 'input.png'))

        if not os.path.exists(os.path.join(subDir, 'Walls.png')) or recreate:
            # Generate mask
            sc = GeomPoint2D(x=floor.image_scale, y=-floor.image_scale)

            mask = np.zeros((h, w, 1), np.uint8)
            mask_pil = Image.new('L', size=(w, h))
            image_draw = ImageDraw.Draw(mask_pil)
            for r in floor.rect:
                new_points = list()
                for p in r.points:
                    new_points.append(p * sc)

                image_draw.polygon([(p.x, p.y) for p in new_points], fill='white', width=1)

            mask_pil.save(os.path.join(subDir, 'Walls.png'))

        if not os.path.exists(os.path.join(subDir, 'mask.png')) or recreate:
            c_ind = (cv2.imread(os.path.join(subDir, 'Walls.png'), 0)).astype(np.uint8)
            mask[c_ind == 255] = 2 + 1  # border
            # flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            # cv2.floodFill(c_ind, flood_mask, (0, 0), 255)
            # mask[cv2.bitwise_not(c_ind) == 255] = 2 + 1  # fill
            # cv2.imwrite(os.path.join(subDir, 'wall_filled.png'), cv2.bitwise_not(c_ind))
            cv2.imwrite(os.path.join(subDir, 'mask.png'), mask)
        if overlay:
            create_overlay('', subDir, 'input.png', 'mask.png', alpha=0.8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract each floorplan from MLSTRUCT-FP dataset and create folder '
                                                 'structure with input and mask image')
    parser.add_argument('--root_path', help='Path to the dataset', default='../../../Datasets/MLSTRUCT-FP_v1')
    parser.add_argument('--overlay', action='store_true', help='Create overlay to show annotation')
    parser.add_argument('--recreate', action='store_true', help='Recreate flag.')

    args = parser.parse_args()
    create_folders_and_masks(args.root_path, args.overlay, args.recreate)

