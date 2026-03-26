_base_ = ['../../base/default_runtime.py', '../../base/default_model.py',
          '../../datasets/cubicasa5k_augment_multi_plans_augment.py']

model_type = 'zeng'
exp_name = 'zeng_combi'
backbone = 'vgg16'
normalize = True
batch_norm = True
