_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/cubicasa5k_augment.py']

model_type = 'unetpp'
exp_name = 'cubi'
backbone = 'efficientnetb2'
filters = [512, 256, 128, 64, 32]
output_activation = 'softmax'
batch_norm = True
up_rates = (2, 2, 2, 2, 2)

loss_functions = ['asym_unified_focal_loss']

batch_size = 2