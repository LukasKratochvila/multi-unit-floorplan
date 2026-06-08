import argparse
import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils import rgb2ind, ind2rgb


# function for conversion between different class sets
def convert_values(matrix_old: np.array, classes_old:list, classes_new:list):
    matrix_new = matrix_old.copy()
    for c in classes_old:
        if c not in classes_new:
            print(f"Class: {c} not in new class list")
        else:
            matrix_new[matrix_old == classes_old.index(c)] = classes_new.index(c)
    return matrix_new



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for correction of model output')
    parser.add_argument('file', help='Path to the file to correct', default='./result/')

    args = parser.parse_args()

    img = Image.open(args.file).convert('RGB')

    classes = ['Background', 'Walls', 'Glass_walls', 'Railings', 'Doors', 'Sliding_doors', 'Windows', 'Stairs_all']
    model_classes = ['Background', 'Walls', 'Railings', 'Doors', 'Windows', 'Stairs_all']

    indexes = rgb2ind(np.array(img))

    indexes_c = convert_values(indexes, model_classes, classes)
    out_img = Image.fromarray(ind2rgb(indexes_c)).convert('RGB')
    name = os.path.join(os.path.dirname(args.file), os.path.basename(args.file).replace("result", "result_corrected"))
    out_img.save(name)




