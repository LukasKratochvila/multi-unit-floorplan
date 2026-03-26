_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/cubicasa5k_augment.py']

model_type = 'unet3p'
exp_name = 'unet3p_cubi_full'
backbone = 'EfficientNetB2'
filters = [32, 64, 128, 256, 512]
output_activation = 'Softmax'
batch_norm = True
hhdc = 5
cam = 3

loss_functions = ['asym_unified_focal_loss', 'AutomaticWeightedLoss']

batch_size = 2
