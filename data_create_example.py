import os
import shutil
import argparse
from tqdm import tqdm
import utils

from datasets.floorplans import create_dataset
from datasets.create_data_mask import create_mask, create_overlay
from datasets.split_dataset import split_dataset

from datasets.augment import augmentation
from datasets.create_heatmap import create_heatmap_from_paths

from datasets.cubicasa.cubi_utils import cubi_select  # ,cubiCasaFix_subDirs, remove_list
from datasets.cubicasa.merge_cubi_labels import merge_cubi_labels
from datasets.cvc_fp_extract import create_folder_structure
from datasets.mlstructfp_extract import create_folders_and_masks

from PIL import ImageShow
from utils import MyViewer


def dataset_create(dataset_name, classes, heatmap_inds, data_source, data_dir, ann_dir, recreate, augment, overlay):
    ImageShow.register(MyViewer(5), 0)

    if 'cubicasa' in data_source:
        input_image_name = 'F1_scaled.png'
    else:
        input_image_name = 'input.png'

    if data_source == 'r3d':
        if recreate:
            # Create subfolders for each sample, but firstly have all samples in one folder
            print('Creating sub folders...')
            files = [f.split('.')[0] for f in os.listdir(data_dir)
                     if f.split('.')[-1] == 'jpg']
            for file in tqdm(files):
                files_to_copy = [fil for fil in os.listdir(data_dir)
                                 if fil.split('_')[0] in [file, file + '.jpg']]
                os.mkdir(os.path.join(data_dir, file))
                for f in files_to_copy:
                    if f.split('.')[-1] == 'jpg':
                        out = 'input.jpg'
                    else:
                        out = f.split('_')[-1]
                    shutil.move(os.path.join(data_dir, f), os.path.join(data_dir, file, out))

        # Creation data description and get all samples
        print('Creating dataset splits...')
        sub_dirs = split_dataset(data_source, input_file=input_image_name, data_dir=data_dir, output_dir=ann_dir,
                                 split_ration=0.8, subfolders_only=os.path.exists(ann_dir) and not recreate)
        # Creation of masks
        masks = [sub_dir for sub_dir in sub_dirs
                 if not os.path.exists(os.path.join(data_dir, sub_dir, 'mask.png'))]
        if masks or recreate:
            print('Creating masks...')
            create_mask(masks, path=data_dir)
        if overlay:
            # Create the overlay
            overs = [file for file in sub_dirs
                     if not os.path.exists(os.path.join(data_dir, file, 'overlay.png'))]
            if overs or recreate:
                print('Creating overlay...')
                for over in tqdm(overs, "Processing"):
                    create_overlay(over, data_dir=data_dir, img_name='input.jpg',
                                   mask_name='mask.png', save=True)
        # Creation data tf records
        if not any([file.split('.')[-1] == 'tfrecords' for file in os.listdir(ann_dir)]) or recreate:
            print('Creating tf records...')
            create_dataset(data_source, type='default', input_dirs=ann_dir, output_dir=ann_dir,
                           data_dir=os.path.dirname(data_dir))

    elif data_source in ['cubicasa5k', 'cubicasa5k_fix']:
        # Creation data description and get all samples and filtering w.r.o. paper
        print('Creating dataset splits...')
        sub_dirs = split_dataset(data_source, input_file=input_image_name, data_dir=data_dir, output_dir=ann_dir,
                                 filtering=None, subfolders_only=os.path.exists(ann_dir) and not recreate)
        # Creation of labels and mask file
        masks = [sub_dir for sub_dir in sub_dirs
                 if not os.path.exists(os.path.join(data_dir, sub_dir, 'mask.png')) or recreate]
        if masks:
            print('Creating masks...')
            pbar = tqdm(masks, "Processing")
            for mask in pbar:
                pbar.set_postfix_str(mask)
                merge_cubi_labels(mask, data_dir=data_dir)
        # Creation data tf records
        if not any([file.split('.')[-1] == 'tfrecords' for file in os.listdir(ann_dir)]) or recreate:
            print('Creating tf records...')
            create_dataset(data_source, type='cubicasa5k', input_dirs=ann_dir, output_dir=ann_dir,
                           data_dir=os.path.dirname(data_dir))
        if overlay:
            # Create the overlay
            overs = [sub_dir for sub_dir in sub_dirs
                     if not os.path.exists(os.path.join(data_dir, sub_dir, 'overlay.png')) or recreate]
            if overs:
                print('Creating overlay...')
                pbar = tqdm(overs, "Processing")
                for over in pbar:
                    pbar.set_postfix_str(over)
                    create_overlay(over, data_dir=data_dir, img_name='F1_original.png',
                                   save=True)

    elif data_source == 'CVC-FP':
        if recreate:
            # Creation sub folders structure and mask files
            print('Creating sub folders and masks...')
            create_folder_structure(data_dir, overlay=overlay)
        # Creation data description
        print('Creating dataset splits...')
        cvc_select = [d for d in os.listdir(data_dir) if not d == 'ImagesGT']  # Remove GT folder
        _ = split_dataset(data_source, input_file=input_image_name, data_dir=data_dir, output_dir=ann_dir,
                          filtering=cvc_select, subfolders_only=os.path.exists(ann_dir) and not recreate)

        # Creation data tf records
        if not any([file.split('.')[-1] == 'tfrecords' for file in os.listdir(ann_dir)]) or recreate:
            print('Creating tf records...')
            create_dataset(data_source, input_dirs=ann_dir, output_dir=ann_dir,
                           data_dir=os.path.dirname(data_dir))

    elif data_source == 'MLSTRUCT-FP_v1':
        data_dir = os.path.join(data_dir, data_source)
        if not os.path.exists(data_dir) or recreate:
            # Creation sub folders structure and mask files
            print('Creating masks...')
            create_folders_and_masks(os.path.dirname(data_dir), overlay=overlay, recreate=recreate)

        # Creation data description
        print('Creating dataset splits...')
        _ = split_dataset(data_source, input_file=input_image_name, data_dir=data_dir, output_dir=ann_dir,
                          subfolders_only=os.path.exists(ann_dir) and not recreate)

        # Creation data tf records
        if not any([file.split('.')[-1] == 'tfrecords' for file in os.listdir(ann_dir)]) or recreate:
            print('Creating tf records...')
            create_dataset(data_source, input_dirs=ann_dir, output_dir=ann_dir,
                           data_dir=os.path.dirname(data_dir))
    else:
        print(f"Dataset: {data_source} is not implemented.")
        exit(1)
    # Create config
    cfg = utils.Config(cfg_dict=dict(dataset_name=dataset_name, dataset_file=data_source,
                                     data_root=ann_dir, classes=classes,
                                     heatmap_inds=heatmap_inds))
    cfg.dump('configs/datasets/' + cfg.dataset_file + '.py')

    if augment:
        # Apply augmentation
        dataset_augment = data_source.lower() + '_augment'
        aug_ann_dir = os.path.join(os.path.dirname(ann_dir), dataset_augment)
        aug_data_dir = os.path.join(os.path.dirname(data_dir), dataset_augment)
        if not os.path.exists(aug_data_dir) or recreate:
            augmentation(data_source, input_dir=os.path.dirname(data_dir), output_dir=aug_data_dir, ann_dir=ann_dir)
        sub_dirs = split_dataset(dataset_augment, input_file=input_image_name, data_dir=aug_data_dir, output_dir=aug_ann_dir,
                                 subfolders_only=os.path.exists(aug_ann_dir) and not recreate)
        if heatmap_inds:
            # Create heatmaps
            heat = [sub_dir for sub_dir in sub_dirs if not any(['heatmap' in file.split('.')[0] for file in os.listdir(
                os.path.join(aug_data_dir, sub_dir))]) or recreate]
            create_heatmap_from_paths(dataset_augment, data_dir=aug_data_dir, sub_dirs=heat, show=False)
        # Create data tf records
        if not any([file.split('.')[-1] == 'tfrecords' for file in os.listdir(aug_ann_dir)]) or recreate:
            create_dataset(dataset_augment, type='default', input_dirs=aug_ann_dir, output_dir=aug_ann_dir,
                           data_dir=os.path.dirname(aug_data_dir))
        if overlay:
            # Create the overlay
            overs = [sub_dir for sub_dir in sub_dirs
                     if not os.path.exists(os.path.join(aug_data_dir, sub_dir, 'overlay.png')) or recreate]
            if overs:
                print('Creating overlay...')
                pbar = tqdm(overs, "Processing")
                for over in pbar:
                    pbar.set_postfix_str(over)
                    create_overlay(over, data_dir=aug_data_dir, img_name=input_image_name,
                                   save=True)
        # Create config
        cfg = utils.Config(cfg_dict=dict(dataset_name=dataset_name, dataset_file=dataset_augment,
                                         data_root=aug_ann_dir, classes=classes,
                                         heatmap_inds=heatmap_inds))
        cfg.dump('configs/datasets/' + cfg.dataset_file + '.py')
    exit(0)


if __name__ == '__main__':
    multi_classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
    cubi_classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
    datasets = ['R3D', 'CubiCasa5k', 'CVC-FP', 'MLSTRUCT-FP']

    parser = argparse.ArgumentParser(description='Data creation example. Use it for generation of tfrecords.')
    # New dataset variables
    parser.add_argument('--dataset_name', default='CubiCasa5k', help='Dataset name. (MLSTRUCT-FP)')
    parser.add_argument('--classes', default=cubi_classes, help='Classes of the new dataset. (multi_class)')
    parser.add_argument('--heatmap_inds', default=[], help='Indexes of heatmap to create. ([])')
    # Folders variables
    parser.add_argument('--data_source', default='cubicasa5k', help='Source data folder name. (MLSTRUCT-FP_v1)')
    parser.add_argument('--data_dir', default='/mnt/hdd/lukas/ensembles/data/annotations/hdd/',
                        help='Data path to process, where data_source folder exist. (/mnt/hdd/lukas/ensembles/data/)')
    parser.add_argument('--ann_dir', default='annotations', help='Path to save tfrecords (annotations)')
    # Flags
    parser.add_argument('-r', action='store_true', help='Recreate all steps flag.')
    parser.add_argument('-o', action='store_true', help='Create overlays.')
    parser.add_argument('-a', action='store_true', help='Apply augmentation.')

    args = parser.parse_args()
    if args.dataset_name not in datasets:
        print(f'Dataset: {args.dataset_name} is not implemented.')
        exit(1)

    out_dir = os.path.abspath(os.path.join(args.ann_dir, args.data_source))
    dat_dir = os.path.join(args.data_dir, args.data_source)
    if not os.path.exists(dat_dir):
        print(f'Data_dir: {args.data_dir} does not include data_source folder: {args.data_source} ')
        exit(1)

    dataset_create(args.dataset_name, args.classes, args.heatmap_inds, args.data_source, dat_dir, out_dir,
                   args.r, args.a, args.o)
