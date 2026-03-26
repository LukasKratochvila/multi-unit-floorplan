import logging
import os
import time
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from training.confusion_matrix import CM

import segmentation_models as sm
from datasets import floorplans
from training.metrics import cm_metrics

import utils
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.disable(logging.WARNING)


def evaluate(config_file):
    sm.set_framework('tf.keras')

    conversion = None
    dataset = config_file.eval_cfg.dataset_file
    if dataset == 'r3d_augment':
        conversion = 'r3d'
    elif dataset == 'cubicasa5k_augment':
        conversion = 'cubicasa5k'

    classes = ['bg'] + config_file.eval_cfg.classes
    model = tf.keras.models.load_model(config_file.train_cfg.log_dir, compile=False)
    tta = config_file.eval_cfg['tta']
    post_processing = config_file.eval_cfg['post_processing']
    if tta:
        print('Fix this!')
        exit(0)
        # unet_model_tta = sm.Xnet(backbone_name=config['backbone'], classes=len(classes), activation='softmax',
        #                          tta=True)
        # unet_model_tta.set_weights(unet_model.get_weights())
        # unet_model = unet_model_tta

    model.compile(run_eagerly=True,
                  metrics=[CM(num_classes=len(classes), post_processing=post_processing, conversion=conversion)])

    backbone = config_file.train_cfg.backbone
    normalize = config_file.train_cfg.normalize
    print("Backbone: {0}, normalize: {1}".format(backbone, normalize))
    test_dataset = floorplans.load_test_data(classes, dataset, normalize=normalize, base_dir=config_file.eval_cfg.data_root, n_upsample=config_file.train_cfg.n_up_sample_block)

    data = test_dataset.batch(1)
    prediction = model.evaluate(data)
    cm = prediction[1]
    timestr = time.strftime("%Y%m%d-%H%M%S")

    #np.savetxt('results/cm' + '_' + config_file.train_cfg.model_type + '_' + config_file.train_cfg.exp_name + '_' + str(tta) + '_' +
    #           str(post_processing) + '_' + timestr + '.txt', cm)

    # Plot confusion matrices
    if config_file.eval_cfg.plot_cm:
        cms = [cm, cm / cm.sum(axis=0, keepdims=True), cm / cm.sum(axis=1, keepdims=True)]
        titles = [
            'Original',
            'What classes are responsible for each classification',
            'How each class has been classified'
        ]
        for t, confusion_matrix in enumerate(cms):
            fig, ax = plt.subplots()
            ax.matshow(confusion_matrix)
            ax.set_title(titles[t])
            ax.set_xlabel('Predicted class')
            ax.set_ylabel('True class')
            ax.xaxis.set_ticks_position('bottom')
            ax.set_xticks(list(range(len(classes))))
            ax.set_yticks(list(range(len(classes))))
            ax.set_xticklabels(classes, rotation='vertical')
            ax.set_yticklabels(classes)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    c = round(confusion_matrix[j, i], 2)
                    ax.text(i, j, str(c), va='center', ha='center')
            config_file.eval_cfg.cm_file = (f'results/cm{str(t)}_{config_file.train_cfg.model_type}_{str(tta)}_'
                                            f'{str(post_processing)}_{timestr}')
            plt.savefig(config_file.eval_cfg.cm_file, bbox_inches='tight')
            plt.show()

    config_file.eval_cfg.metrics_file = (f'results/'
                                         f'metrics_{config_file.train_cfg.model_type}_{config_file.train_cfg.exp_name}_'
                                         f'{str(tta)}_{str(post_processing)}_{timestr}.txt')
    # Compute metrics
    f = open(config_file.eval_cfg.metrics_file, 'w')
    table = []
    columns = cm_metrics(cm)

    table.append(['Accuracy', np.diag(cm).sum() / cm.sum()])
    cm_nobg = np.copy(cm)
    cm_nobg[0] = 0
    table.append(['Accuracy no bg', np.diag(cm_nobg).sum() / cm_nobg.sum()])
    for i, c in enumerate(classes):
        table.append(
            [c] + [column[i] for column in columns[:-1]] + [np.array([metrics[i] for metrics in columns[-1]]) / sum(
                [metrics[i] for metrics in columns[-1]])])
    table.append(['Mean'] + [np.nanmean(column) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1], axis=1)) / np.sum(columns[-1], axis=1).sum()])
    table.append(['Mean no bg'] + [np.nanmean(column[1:]) for column in columns[:-1]] + [
        np.array(np.sum(columns[-1][1:], axis=1)) / np.sum(columns[-1][1:], axis=1).sum()])
    f.write(
        tabulate(table, headers=['Class', 'Class accuracy', 'Recall', 'Precision', 'F1', 'IoU', 'fwRecall', 'fwIoU',
                                 'TP, FP, TN, FN']))
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate trained model on test dataset')
    parser.add_argument('config', help='Config file path (could be multiple)', nargs='+')
    parser.add_argument('--dataset', default='', help='Dataset config from ./config/datasets folder')
    parser.add_argument('--plot_cm', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--post', action='store_true', help='Process post processing')
    parser.add_argument('--tta', action='store_true', help='Process tta')

    args = parser.parse_args()
    configs = args.config
    for config in configs:
        print('Processing config: ', config)
        # load train_cfg
        cfg = utils.Config.fromfile(config)
        # load dataset info
        cfg.eval_cfg = dict()
        cfg.eval_cfg.dataset_file = cfg.train_cfg.dataset_file if args.dataset == '' else args.dataset
        dsets_cfg = utils.Config.fromfile(os.path.join('./configs/datasets', f'{cfg.eval_cfg.dataset_file}.py'))
        cfg['eval_cfg'] = dsets_cfg.to_dict()
        # prepare variables
        cfg.eval_cfg.plot_cm = args.plot_cm
        cfg.eval_cfg.post_processing = args.post
        cfg.eval_cfg.tta = args.tta
        tic = time.time()
        evaluate(cfg)
        toc = time.time()
        print('total evaluation time = {} minutes'.format((toc - tic) / 60))

        cfg.eval_cfg.eval_time = toc - tic
        cfg.dump(os.path.join(cfg.train_cfg.log_dir, f'evaluate_cfg_{time.strftime("%Y%m%d-%H%M%S")}.py'))
    print('============== End ==============')
