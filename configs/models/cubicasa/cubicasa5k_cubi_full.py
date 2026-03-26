_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/cubicasa5k_augment.py']

model_type = 'cubicasa5k'
exp_name = 'cubicasa5k_cubi_full'
backbone = 'vgg16'
normalize = True
batch_norm = True
