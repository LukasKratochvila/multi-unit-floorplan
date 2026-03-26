# runtime_cfg
local_rank = 0
epochs = 200
batch_size = 10
train_buffer_size = 400
data_reduction = None

## Loss
loss_functions = ['balanced_entropy']  # categorical_crossentropy, asym_unified_focal_loss,
# heatmap_regression_loss, heatmap_regression_loss_nomean, adaptive_affinity_loss
optimizer = dict(type='Adam', learning_rate=1e-4, lossScale=False)
metrics = ['CategoricalAccuracy']

## Model callbacks
checkpoint_weights_only = False
run_eagerly = False
verbose = 2
