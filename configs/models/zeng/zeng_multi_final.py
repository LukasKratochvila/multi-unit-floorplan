_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/multi_plans_augment.py']

model_type = 'zeng'
exp_name = 'zeng_multi_final'
backbone = 'vgg16'
normalize = True
batch_norm = True
