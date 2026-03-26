import random
import os

from typing import List

heatmap_classes = {
    'r3d_augment': ['openings'],
    'cubicasa5k_augment': ['doors', 'windows'],
    'multi_plans_augment': ['doors', 'sliding_doors', 'windows'],
}


def split_dataset(dataset, input_file: str = 'input.png', data_dir: str = 'annotations/hdd/',
                  output_dir: str = './', split_ration: float = 0.6, filtering: List = None,
                  test_only: bool = False, subfolders_only: bool = False) -> List[str]:
    """
    Function for creation tran, val, test sets descriptions

    :param dataset: Name of new dataset
    :param input_file: Filename used for processing as input
    :param data_dir: Folder with input data
    :param output_dir: Folder where create description
    :param split_ration: Ration between train and (val, test), e.g. 0.6 means train sets includes 0.6% of all data
    :param filtering: Subset of folders which will be use
    :param test_only: If true - All data will be in test subset
    :param subfolders_only: If true - Not create description, only returns subfolders

    :return: returns list of subfolders in dataset
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    heatmaps = heatmap_classes.get(dataset, [])

    if dataset == 'cubicasa':
        sub_dirs = []
        for file in ["train.txt", "test.txt", "val.txt"]:
            sub_dirs += open(os.path.join(data_dir, file), 'r').read().splitlines()
        sub_dirs = [d[1:] for d in sub_dirs]
        if filtering:
            sub_dirs = [d for d in sub_dirs if d.split('/')[-2] in filtering]
            filtering = None
    else:
        sub_dirs = os.listdir(data_dir)
    if filtering:
        sub_dirs = [d for d in sub_dirs if d in filtering]

    if subfolders_only:
        print("Sub folders only: Dataset: {0} Total samples: {1}".format(dataset, len(sub_dirs)))
        return sub_dirs

    random.shuffle(sub_dirs)

    if test_only:
        n_train = 0
        n_val = 0
        n_test = len(sub_dirs)
    else:  # Create split: train/(val, test)
        n_train = int(split_ration * len(sub_dirs))
        n_val = int((len(sub_dirs) - n_train) / 2)
        n_test = len(sub_dirs) - n_train - n_val

    print("Total: {0}, train: {1}, val: {2}, test: {3}".format(len(sub_dirs), n_train, n_val, n_test))

    train = sub_dirs[:n_train]
    val = sub_dirs[n_train:n_train + n_val]
    test = sub_dirs[n_train + n_val:]

    for dir, type in zip([train, val, test], ['train', 'val', 'test']):
        f = open(os.path.join(output_dir, dataset + '_' + type + '.txt'), 'w')
        for i in dir:
            input = os.path.join(dataset, i, input_file)
            mask = os.path.join(dataset, i, 'mask.png')
            line = input + '\t' + mask
            for h in heatmaps:
                line += '\t' + os.path.join(dataset, i, 'heatmap_' + h + '.png')
            f.write(line + '\n')
        f.close()
    return sub_dirs


if __name__ == '__main__':
    # split_dataset('cubicasa5k_single_augment')
    split_dataset('cubicasa5k_augment')
    # split_dataset('multi_plans', 'image.jpg')
    # split_dataset('multi_plans_augment')
    # split_dataset('multi_plans_test', 'image.jpg', True)
    # split_dataset('multi_plans_test_augment', test_only=True)
    split_dataset('r3d_augment')
