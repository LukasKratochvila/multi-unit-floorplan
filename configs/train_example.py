# Necessary fields for training
_base_ = ['base/default_runtime.py', 'base/default_model.py', 'datasets/r3d_augment.py']

## Train specs
exp_name = 'Example_unetpp'

## Model specs
model_type = 'unetpp'
backbone = 'efficientnetb5'
output_activation = 'softmax'
filters = [512, 256, 128, 64, 32]
up_rates = (2, 2, 2, 2, 2)
n_up_sample_block = len(up_rates)
batch_norm = True

## Runtime specs
batch_size = 1
epochs = 1
