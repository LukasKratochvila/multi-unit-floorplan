import logging
import os
import time

import tensorflow as tf
from tensorflow.python.keras.mixed_precision.loss_scale_optimizer import LossScaleOptimizer
#from tensorflow.python.keras.mixed_precision.experimental.loss_scale_optimizer import LossScaleOptimizer
#from tensorflow.python.training.experimental import mixed_precision

import segmentation_models as sm
from datasets import floorplans
from segmentation_models.models import zeng, r2v
from segmentation_models.models.cab1.cab1 import cab1
from segmentation_models.models.cab2.cab2 import cab2
from segmentation_models.models.ours_multi.ours_multi import ours_multi
from segmentation_models.models.unet3plus.model_unet_2d import unet_2d
from segmentation_models.models.unet3plus.model_unet_3plus_2d import unet_3plus_2d
from training import Trainer, loss_functions
from training.AutomaticWeightedLoss import AutomaticWeightedLoss, AutomaticWeightedLossCallback

from tqdm.keras import TqdmCallback
from utils import Config, mkdir_or_exist, get_args_dict

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.disable(logging.WARNING)


def main():
    sm.set_framework('tf.keras')

    # Get default options
    help_cfg = Config.fromfile('configs/base/train_ops_help.py')
    # Parse args
    parser = help_cfg.auto_argparser('Train a model options')

    # Process configs
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args_dict = get_args_dict(args)

    configs = args_dict.pop('config')
    if args_dict.get('work_dir', None) is None:
        train_folder = args_dict.get('exp_name', None)
        if train_folder is None:
            args_dict['work_dir'] = os.path.abspath('./')
        else:
            # create sub folder train_folder
            args_dict['work_dir'] = os.path.abspath(
                os.path.join('./', os.path.splitext(os.path.basename(train_folder))[0]))
    # create work_dir
    mkdir_or_exist(args_dict['work_dir'])
    # Run for each config
    for config in configs:
        cfg = Config.fromfile(config)
        cfg.merge_from_dict(args_dict)

        # Print warning for options which are not specified
        for key in help_cfg.to_dict().keys():
            if key not in cfg.to_dict().keys():
                print(f'Warning: option {key} is not in config.')

        tic = time.time()
        train(cfg)
        toc = time.time()

        cfg.training_time = toc - tic

        cfg = Config(cfg_dict=dict(train_cfg=cfg.to_dict()))
        cfg.dump(os.path.join(cfg.train_cfg.log_dir, 'training_cfg.py'))

        print('total training time = {} minutes'.format((toc - tic) / 60))
        print()
        print('Waiting 1 min')
        time.sleep(60)
        print('Resuming')
        print()
    print('Finished')


def train(config):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    exp_name = config.get('exp_name', '')
    # Parameters
    model_type = config.get('model_type', None)
    backbone = config.get('backbone', None)
    resume_from = config.get('resume_from', None)
    deep_supervision = config.get('deep_supervision', False)
    filters = config.get('filters', [])
    up_rates = config.get('up_rates', [])
    n_up_sample_block = config.get('n_up_sample_block', None)
    output_activation = config.get('output_activation', None)
    backbone_weights = config.get('backbone_weights', None)
    hhdc = config.get('hhdc', None)
    cam = config.get('cam', None)
    aaf = config.get('aaf', [])
    aaf_count = len(aaf)
    w_edge = None
    w_not_edge = None
    baseline = config.get('baseline', False)
    batch_norm = config.get('batch_norm', False)

    classes = ['bg'] + config.get('classes', [])
    heatmap_inds = config.get('heatmap_inds', [])
    dataset = config.get('dataset_file', None)
    normalize = config.get('normalize', False)
    data_root = config.get('data_root', None)
    data_reduction = config.get('data_reduction', None)

    epochs = config.get('epochs', 1)
    batch_size = config.get('batch_size', 1)
    train_buffer_size = config.get('train_buffer_size', 1)
    checkpoint_weights_only = config.get('checkpoint_weights_only', False)
    loss_function = config.get('loss_functions', [])
    optimizer = config.get('optimizer', None)
    cfg_metrics = config.get('metrics', [])
    run_eagerly = config.get('run_eagerly', False)
    verbose = config.get('verbose', 0)

    if not dataset:
        ValueError('Dataset must be specify!')
    if not os.path.exists(data_root):
        ValueError(f'Data dir: {data_root} does not exist!')
    if not optimizer:
        ValueError('Optimizer must be specify!')
    if model_type not in ['zeng', 'cubicasa5k', 'unet', 'unetpp', 'unet3p', 'ours_multi', 'cab1', 'cab2']:
        ValueError(f'Model: {model_type} is not implemented!')

    mkdir_or_exist(os.path.join(config.work_dir, 'models'))
    config.log_dir = os.path.join(config.work_dir, 'models', '_'.join(
        [model_type, exp_name, str(backbone or ''), ','.join(map(str, filters)), dataset,
         time.strftime("%Y%m%d-%H%M%S")]))

    with strategy.scope():
        metrics = []
        for m in cfg_metrics:
            if m == 'CategoricalAccuracy':
                metrics.append(tf.metrics.CategoricalAccuracy())
            else:
                ValueError(f'Metric: {m} is not Implemented!')

        if optimizer.get('type', None) == 'Adam':
            loss_scale = optimizer.get('lossScale', None)
            optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer.get('learning_rate', 1e-4))
            if loss_scale:
                optimizer = LossScaleOptimizer(optimizer, loss_scale='dynamic')
        else:
            ValueError(f'Not implemented optimizer: {optimizer}')

        if not loss_function:
            ValueError('Loss function must be specify!')
        if len(loss_function) == 1:
            lf = loss_function[0]
            if lf == 'balanced_entropy':
                loss_function = loss_functions.balanced_entropy(len(classes))
            elif lf == 'categorical_crossentropy':
                loss_function = loss_functions.categorical_crossentropy(len(classes))
            if lf == 'asym_unified_focal_loss':
                loss_function = loss_functions.asym_unified_focal_loss(len(classes))
            else:
                ValueError(f'Not implemented loss function: {lf}')
        else:
            loss_funcs, names, inds, dec = [], [], [], []
            for lf in loss_function:
                if lf == 'asym_unified_focal_loss':
                    loss_funcs.append(loss_functions.asym_unified_focal_loss(len(classes)))
                    names.append(lf)
                    inds.append(0)
                    dec.append(False)
                if lf == 'heatmap_regression_loss_nomean':
                    loss_funcs.append(
                        loss_functions.heatmap_regression_loss_nomean(len(classes), heatmap_inds))
                    names.append(lf)
                    inds.append(len(inds))
                    dec.append(False)
                if lf == 'heatmap_regression_loss':
                    loss_funcs.append(
                        loss_functions.heatmap_regression_loss(len(classes), heatmap_inds))
                    names.append(lf)
                    inds.append(len(inds))
                    dec.append(False)
                if lf == 'adaptive_affinity_loss':
                    if aaf_count > 0:
                        init_w = tf.constant_initializer(1 / aaf_count)
                        w_edge = tf.Variable(
                            name='edge_w',
                            initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                            dtype=tf.float32,
                            trainable=True)
                        w_not_edge = tf.Variable(
                            name='nonedge_w',
                            initial_value=init_w(shape=(1, 1, 1, len(classes), 1, aaf_count)),
                            dtype=tf.float32,
                            trainable=True)
                        aaf_ind = len(inds)
                        for a, s in enumerate(aaf):
                            loss_funcs.append(
                                loss_functions.adaptive_affinity_loss(size=s, k=a, num_classes=len(classes),
                                                                      w_edge=w_edge, w_not_edge=w_not_edge))
                            names.append('AAF({0}x{0})'.format(2 * s + 1))
                            inds.append(aaf_ind)
                            dec.append(True)
            loss_function = AutomaticWeightedLoss(loss_funcs, names, inds, dec, epochs, config.log_dir)

        if resume_from:
            print('Resuming training')
            custom_objects = {'loss_function': loss_function}
            unet_model = tf.keras.models.load_model(resume_from, custom_objects=custom_objects)
        else:
            if model_type == 'zeng':
                unet_model = zeng.deepfloorplanModel(classes)
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'cubicasa5k':
                unet_model = r2v.hg_furukawa_original(len(classes))
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unet':
                unet_model = unet_2d((None, None, 3), n_labels=len(classes), backbone=backbone,
                                     filter_num=filters, output_activation=output_activation, batch_norm=batch_norm,
                                     weights=backbone_weights, aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                # Apply loss scaling for optimizer
                unet_model.compile(loss=loss_function.combined_loss(), optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unetpp':
                unet_model = sm.Xnet(backbone_name=backbone, classes=len(classes), decoder_filters=filters,
                                     activation=output_activation, encoder_weights=backbone_weights,
                                     upsample_rates=up_rates)
                unet_model.compile(loss=loss_function, optimizer=optimizer,
                                   metrics=metrics, run_eagerly=run_eagerly)
            elif model_type == 'unet3p':
                unet_model = unet_3plus_2d((None, None, 3), n_labels=len(classes), backbone=backbone,
                                           filter_num_down=filters, output_activation=output_activation,
                                           batch_norm=batch_norm, weights=backbone_weights, aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge

                # Apply loss scaling for optimizer
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'ours_multi':
                unet_model = ours_multi((None, None, 3), n_labels=len(classes), backbone=backbone,
                                        filter_num_down=filters,
                                        output_activation=output_activation,
                                        batch_norm=batch_norm,
                                        weights=backbone_weights,
                                        aaf=(aaf_count > 0))

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'cab1':
                unet_model = cab1((None, None, 3), n_labels=len(classes), backbone=backbone,
                                  filter_num_down=filters,
                                  output_activation=output_activation,
                                  batch_norm=batch_norm,
                                  deep_supervision=deep_supervision,
                                  weights=backbone_weights,
                                  aaf=(aaf_count > 0),
                                  use_hhdc=hhdc,
                                  use_cam=cam)

                unet_model.automatic_loss = loss_function
                unet_model.loss_sigmas = loss_function.sigmas
                if aaf_count > 0:
                    unet_model.w_edge = w_edge
                    unet_model.w_not_edge = w_not_edge
                unet_model.compile(loss=loss_function.combined_loss(),
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   run_eagerly=run_eagerly)
            elif model_type == 'cab2':
                if baseline:
                    unet_model = cab2((None, None, 3), n_labels=len(classes), backbone=backbone,
                                      filter_num_down=filters,
                                      output_activation=output_activation,
                                      batch_norm=batch_norm,
                                      deep_supervision=deep_supervision,
                                      weights=backbone_weights,
                                      aaf=(aaf_count > 0),
                                      use_hhdc=hhdc,
                                      use_cam=cam)
                    unet_model.compile(loss=loss_function,
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       run_eagerly=run_eagerly)
                else:
                    unet_model = cab2((None, None, 3), n_labels=len(classes), backbone=backbone,
                                      filter_num_down=filters,
                                      output_activation=output_activation,
                                      batch_norm=batch_norm,
                                      deep_supervision=deep_supervision,
                                      weights=backbone_weights,
                                      aaf=(aaf_count > 0),
                                      use_hhdc=hhdc,
                                      use_cam=cam)

                    unet_model.automatic_loss = loss_function
                    unet_model.loss_sigmas = loss_function.sigmas
                    if aaf_count > 0:
                        unet_model.w_edge = w_edge
                        unet_model.w_not_edge = w_not_edge
                    unet_model.compile(loss=loss_function.combined_loss(),
                                       optimizer=optimizer,
                                       metrics=metrics,
                                       run_eagerly=run_eagerly)

        print('Used config: ', config)
        train_dataset, validation_dataset, test_dataset = floorplans.load_train_data(classes, dataset,
                                                                                     normalize=normalize,
                                                                                     buffer_size=train_buffer_size,
                                                                                     base_dir=data_root,
                                                                                     n_upsample=n_up_sample_block,
                                                                                     reduction_ratio=data_reduction)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=verbose)
        progbar = TqdmCallback(verbose=2)

        callbacks = [early_stop, progbar]
        if model_type in ['unetpp', 'unet3p', 'ours_multi', 'cab1', 'cab2']:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5,
                                                         verbose=verbose)
            callbacks.append(reduce_lr)
        if hasattr(unet_model, 'automatic_loss'):
            callbacks.append(AutomaticWeightedLossCallback(aaf_count > 0))
        trainer = Trainer(checkpoint_callback=True, checkpoint_weights_only=checkpoint_weights_only,
                          learning_rate_scheduler=None, tensorboard_images_callback=False, callbacks=callbacks,
                          log_dir_path=config.log_dir)
        trainer.fit(unet_model,
                    train_dataset,
                    validation_dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=verbose)

    del unet_model
    tf.keras.backend.clear_session()
    print('========== DONE ==========')


if __name__ == "__main__":
    main()
