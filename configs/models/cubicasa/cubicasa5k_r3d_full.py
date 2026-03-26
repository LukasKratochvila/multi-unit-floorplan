_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/r3d_augment.py']

model_type = 'cubicasa5k'
exp_name = 'cubicasa5k_r3d_full'
backbone = 'vgg16'
normalize = True
batch_norm = True
