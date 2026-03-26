# Default training ops only for help (have to be included in config)
_base_ = ['default_runtime.py', 'default_model.py']

train_folder = 'Folder_for_training_experiments.'
work_dir = 'The_dir_to_save_logs_and_models'  # Not necessary

## Dataset specs
dataset_file = 'Dataset_file_to_open'
data_root = 'Where_to_find_the_tfrecords'
classes = ['Dataset_classes']
heatmap_inds = [0]  # opening indexes
data_reduction = 0

## Model specs
model_type = 'Model_name'  # ['zeng', 'cubicasa5k', 'unet', 'unetpp', 'unet3p', 'ours_multi', 'cab1', 'cab2']
exp_name = 'Experiment_name'
backbone = 'Backbone_name'
filters = [0, 0]
up_rates = (0, 0)
output_activation = 'Activation_function'  # 'Softmax' 'softmax'
backbone_weights = 'Weight_for_backnbone'  # 'imagenet'
aaf = [0]
hhdc = 0
cam = 0
n_up_sample_block = 0
load_from = 'The_checkpoint_file_to_load_weights_from'
resume_from = 'The_checkpoint_file_to_resume_from'

## Runtime specs
# Loss
loss_functions = ['Loss_functions']  # categorical_crossentropy, asym_unified_focal_loss,
# heatmap_regression_loss, heatmap_regression_loss_nomean, adaptive_affinity_loss
optimizer = dict(type='Type_of_optimazer', learning_rate=1e-4, lossScale=False)
metrics = ['Metrics_to_evaluate']


