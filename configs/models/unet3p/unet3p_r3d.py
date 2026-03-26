_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/r3d_augment.py']

model_type = 'unet3p'
exp_name = 'r3d'
backbone = 'EfficientNetB2'
filters = [32, 64, 128, 256, 512]
output_activation = 'Softmax'
batch_norm = True

loss_functions = ['asym_unified_focal_loss', 'AutomaticWeightedLoss']

batch_size = 2
