# model_cfg
model_type = None  # ['zeng', 'cubicasa5k', 'unet', 'unetpp', 'unet3p', 'ours_multi', 'cab1', 'cab2']
exp_name = None
backbone = None
filters = []  # [32, 64, 128, 256, 512] [512, 256, 128, 64, 32]
up_rates = None
output_activation = None  # 'Softmax' 'softmax'
backbone_weights = 'imagenet'  # 'imagenet'
aaf = []
hhdc = False
cam = False
deep_supervision = False
baseline = False
normalize = False
n_up_sample_block = None
batch_norm = False
load_from = None
resume_from = None
