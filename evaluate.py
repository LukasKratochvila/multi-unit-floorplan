import logging
import os
import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

from training.confusion_matrix import CM

import segmentation_models as sm
from datasets import floorplans
from training.metrics import cm_metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.disable(logging.WARNING)


def main():
    sm.set_framework('tf.keras')

    configs = [
        ################ R3D runs ################
        # {
        #     'model': 'zeng_zeng_r3d_final_vgg16_32,64,128,256,512_r3d_augment_20220809-184713',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment']
        # },
        # {
        #     'model': 'unet3p_unet3p_gn_EfficientNetB2_32,64,128,256,512_r3d_augment_20220809-165907',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'cab2_map_abc_r3d_EfficientNetB2_32,64,128,256,512_r3d_augment_20220809-140528',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'cab2_map_abc_final1x1_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220811-144349',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },

        ################ CubiCasa runs ################
        # {
        #     'model': 'zeng_zeng_cubi_final_vgg16_64,128,256,512,1024_cubicasa5k_augment_20220725-194943',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        #     'post_processing': False
        # },
        # {
        #     'model': 'unet3p_unet3p_cubi_final_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220812-081830',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        #     'tta': False
        # },
        # {
        #     'model': 'cab2_map_abc_cubi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220811-200057',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        #     'tta': False
        # },

        ################ U-Net++ runs ################
        # {
        #     'model': 'unetpp_r3d_full_efficientnetb2_32,64,128,256,512_r3d_augment_20220815-145449',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'unetpp_cubi_full_efficientnetb2_32,64,128,256,512_cubicasa5k_augment_20220815-183455',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
        # {
        #     'model': 'unetpp_multi_full_efficientnetb2_32,64,128,256,512_multi_plans_augment_20220815-235150',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['multi_plans_augment'],
        # },
        # {
        #     'model': 'unetpp_combi_efficientnetb2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220816-162654',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        ################ U-Net runs ################
        # {
        #     'model': 'unet_r3d_full2_EfficientNetB2_32,64,128,256,512_r3d_augment_20220820-042826',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'unet_cubi2_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220820-051332',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
        # {
        #     'model': 'unet_multi2_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220820-071150',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['multi_plans_augment'],
        # },
        # {
        #     'model': 'unet_combi2_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220820-083922',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        ################ Arch 1 runs ################
        # {
        #     'model': 'cab1_map_abc_r3d_EfficientNetB2_32,64,128,256,512_r3d_augment_20220817-184743',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'cab1_map_abc_cubi_full_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220817-222735',
        #     'name': 'pp',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
        # {
        #     'model': 'cab1_map_abc_regular_smaller_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220813-122712',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['multi_plans_augment'],
        # },
        # {
        #     'model': 'cab1_map_abc_combi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220817-094140',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        ################ DS runs ################
        {
            'model': 'cab2_map_abc_ds_multi_heatmap_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220819-214610',
            'name': 'billinear',
            'backbone': 'efficientnetb2',
            'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
            'datasets': ['multi_plans_augment'],
        },
        {
            'model': 'cab2_map_abc_ds_multi_heatmap_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220819-171101',
            'name': 'nearest',
            'backbone': 'efficientnetb2',
            'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
            'datasets': ['multi_plans_augment'],
        },

        ################ Abl AM runs ################
        # {
        #     'model': 'cab2_map_abc_single_sam_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220819-001613',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },
        # {
        #     'model': 'cab2_map_abc_single_cam_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220818-141119',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        ################ Abl task runs ################
        # {
        #     'model': 'cab2_map_abc_combi_baseline_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220820-112540',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        ################ CubiCasa5K runs ################
        # {
        #     'model': 'cubicasa5k_cubicasa5k_r3d_full_vgg16_32,64,128,256,512_r3d_augment_20220814-165742',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'openings'],
        #     'datasets': ['r3d_augment'],
        # },
        # {
        #     'model': 'cubicasa5k_cubicasa5k_cubi_full_vgg16_32,64,128,256,512_cubicasa5k_augment_20220814-181509',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
        # {
        #     'model': 'cubicasa5k_cubicasa5k_multi_full_vgg16_32,64,128,256,512_multi_plans_augment_20220814-141054',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['multi_plans_augment'],
        # },
        # {
        #     'model': 'cubicasa5k_combi_efficientnetb2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220816-233440',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },
        # {
        #     'model': 'cubicasa5k_combi_vgg16_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220819-101935',
        #     'name': '',
        #     'backbone': 'vgg16',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment', 'multi_plans_augment'],
        # },

        # {
        #     'model': 'unet3p_gn_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-091658',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all'],
        #     'datasets': ['multi_plans_augment'],
        # },

        # {
        #     'model': 'unet3p_gn_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220801-182737',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
        # {
        #     'model': 'unet3p_gn_aaf24_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220802-003016',
        #     'name': '',
        #     'backbone': 'efficientnetb2',
        #     'classes': ['walls', 'railings', 'doors', 'windows', 'stairs_all'],
        #     'datasets': ['cubicasa5k_augment'],
        # },
    ]

    for config in configs:
        conversion = None
        datasets = config['datasets']
        if len(datasets) == 1:
            if datasets[0] == 'r3d_augment':
                conversion = 'r3d'
            elif datasets[0] == 'cubicasa5k_augment':
                conversion = 'cubicasa5k'

        classes = ['bg'] + config['classes']
        unet_model = tf.keras.models.load_model('models/' + config['model'], compile=False)
        tta = 'tta' in config and config['tta']
        post_processing = 'post_processing' in config and config['post_processing']
        if tta:
            print('Fix this!')
            exit(0)
            # unet_model_tta = sm.Xnet(backbone_name=config['backbone'], classes=len(classes), activation='softmax',
            #                          tta=True)
            # unet_model_tta.set_weights(unet_model.get_weights())
            # unet_model = unet_model_tta

        unet_model.compile(run_eagerly=True, metrics=[CM(num_classes=len(classes), post_processing=post_processing, conversion=conversion)])

        # NORMALIZE=False IF EFFICIENTNET IS USED
        # Otherwise=True
        backbone = config['backbone']
        normalize = backbone[0] != 'e'
        print("Backbone: {0}, normalize: {1}".format(backbone, normalize))
        test_dataset = floorplans.load_test_data(classes, config['datasets'], normalize=normalize)

        data = test_dataset.batch(1)
        eval = unet_model.evaluate(data)
        cm = eval[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")

        np.savetxt(
            'results/cm' + '_' + config['model'] + '_' + config.get('name', '') + '_' + str(tta) + '_' + str(
                post_processing) + '_' + timestr + '.txt',
            cm)

        # Plot confusion matrices
        plot_cm = False
        if plot_cm:
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
                plt.savefig('results/cm' + str(t) + '_' + config['model'] + '_' + str(tta) + '_' + str(
                    post_processing) + '_' + timestr, bbox_inches='tight')
                plt.show()

        # Compute metrics
        f = open('results/metrics' + '_' + config['model'] + '_' + config.get('name', '') + '_' + str(tta) + '_' + str(
            post_processing) + '_' + timestr + '.txt', 'w')
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
    tic = time.time()
    main()
    toc = time.time()
    print('total evaluation time = {} minutes'.format((toc - tic) / 60))
