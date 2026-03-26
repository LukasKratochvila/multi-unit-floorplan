_base_ = ['../../base/default_runtime.py', '../../base/default_model.py',
          '../../datasets/cubicasa5k_augment_multi_plans_augment.py']

model_type = 'cubicasa5k'
exp_name = 'cubicasa5k_combi'
backbone = 'efficientnetb2'
batch_norm = True
