_base_ = ['../../base/default_runtime.py', '../../base/default_model.py', '../../datasets/r3d_augment.py']

model_type = 'cab2'
exp_name = 'map_abc_r3d_heatmap_aaf'
backbone = 'EfficientNetB2'
filters = [32, 64, 128, 256, 512]
output_activation = 'Softmax'
batch_norm = True
aaf = [2, 4]
hhdc = 5
cam = 3

loss_functions = ['asym_unified_focal_loss', 'heatmap_regression_loss', 'adaptive_affinity_loss', 'AutomaticWeightedLoss']

batch_size = 2
