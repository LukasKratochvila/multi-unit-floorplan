# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings

import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling2D

import numpy as np

from keras_applications import imagenet_utils
from keras_applications import get_submodules_from_kwargs

backend = None
layers = None
models = None
keras_utils = None

def build_norm_layer(norm_layer, name, eps=1e-6):
    if norm_layer == 'BN':
        return layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-05, name=name)
    elif norm_layer == 'LN':
        return layers.LayerNormalization(axis=-1, epsilon=eps, name=name)
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')


def build_act_layer(act_layer, name):
    if act_layer == 'ReLU':
        return layers.ReLU(name=name)
    elif act_layer == 'SiLU':
        return SiLU(name=name)
    elif act_layer == 'GELU':
        return GELU(name=name)

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class SiLU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs * backend.sigmoid(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        return 0.5 * inputs * (1 + tf.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * tf.pow(inputs,3))))

    def compute_output_shape(self, input_shape):
        return input_shape


def preprocess_input(x, **kwargs):
    return x


def InternImage(include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                batch_size=None,
                num_classes=1000,
                channels=64,
                depths=[4, 4, 18, 4],
                groups=[4, 8, 16, 32],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.2,
                drop_path_type='linear',
                act_layer='GELU',
                norm_layer='LN',
                layer_scale=None,
                offset_scale=1.0,
                post_norm=False,
                cls_scale=1.5,
                with_cp=False,
                dw_kernel_size=None,  # for InternImage-H/G
                use_clip_projector=False,  # for InternImage-H/G
                level2_post_norm=False,  # for InternImage-H/G
                level2_post_norm_block_ids=None,  # for InternImage-H/G
                res_post_norm=False,  # for InternImage-H/G
                center_feature_scale=False,  # for InternImage-H/G
                out_indices=(0, 1, 2, 3),
                init_cfg=None,
                pooling=None,
                **kwargs):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    num_levels = len(depths)

    num_features = int(channels * 2**(num_levels - 1))
    #logger = logging.getLogger()
    #logger.info(f'using core type: {core_op}')
    #logger.info(f'using activation layer: {act_layer}')
    #logger.info(f'using main norm layer: {norm_layer}')
    #logger.info(f'using dpr: {drop_path_type}, {drop_path_rate}')
    #logger.info(f"level2_post_norm: {level2_post_norm}")
    #logger.info(f"level2_post_norm_block_ids: {level2_post_norm_block_ids}")
    #logger.info(f"res_post_norm: {res_post_norm}")

    input_shape = imagenet_utils._obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    #if None in input_shape and input_tensor is None:
    #    raise ValueError('The `input_shape` argument should not has None item. '
    #                     'The model cannot be created. ')
    #if batch_size is None:
    #    batch_size = 1
    #input_shape = (batch_size,) + input_shape

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = StemLayer(img_input,
                  name="stem",
                  out_chans=channels,
                  act_layer=act_layer,
                  norm_layer=norm_layer)
    x = layers.Dropout(rate=drop_rate, name="pos_drop")(x)

    dpr = tf.linspace(0., drop_path_rate, sum(depths)).numpy().tolist()

    if drop_path_type == 'uniform':
        for i in range(len(dpr)):
            dpr[i] = drop_path_rate

    seq_out = []
    for i in range(num_levels):
        post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
            i == 2) else None # for InternImage-H/G
        x, x_ = InternImageBlock(x,
            name="level"+str(i),
            channels=int(channels * 2**i),
            depth=depths[i],
            groups=groups[i],
            return_wo_downsample=True,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
            act_layer=act_layer,
            norm_layer=norm_layer,
            post_norm=post_norm,
            downsample=(i < num_levels - 1),
            layer_scale=layer_scale,
            offset_scale=offset_scale,
            with_cp=with_cp,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            post_norm_block_ids=post_norm_block_ids, # for InternImage-H/G
            res_post_norm=res_post_norm, # for InternImage-H/G
            center_feature_scale=center_feature_scale # for InternImage-H/G
        )
        if i in out_indices:
            seq_out.append(x_) # segmentation output

        if not use_clip_projector:  # for InternImage-T/S/B/L/XL
            conv_head = layers.Conv2D(int(num_features * cls_scale), strides=1,
                          kernel_size=1, use_bias=False, name="conv_head")(x)
            conv_head = build_norm_layer('BN', name="conv_head_norm")(conv_head)
            output = build_act_layer(act_layer, name="conv_head_act")(conv_head)
            if include_top:
                output = AdaptiveAveragePooling2D((1, 1), name="avgpool")(output)
                output = layers.Flatten(name="flatten")(output)
                if num_classes > 0:
                    output = layers.Dense(num_classes, name="head")(output)
        else:  # for InternImage-H/G
            #pretrain_embed_dim, _stride, attnpool_num_heads, clip_embed_dim = 1024, 2, 16, 768
            #dcnv3_head_x4 = layers.Conv2D(out_channels=pretrain_embed_dim * (_stride ** 2),
            #                              kernel_size=1, name="dcnv3_head_x4")(seq_out[-1])

            #dcnv3_head_x4 = layers.PixelShuffle(_stride)(dcnv3_head_x4)
            #dcnv3_head_x3 = layers.Conv2D(out_channels=pretrain_embed_dim,
            #                              kernel_size=1, name="dcnv3_head_x3")
            #clip_projector = AttentionPoolingBlock(
            #    dim=pretrain_embed_dim,
            #    num_heads=attnpool_num_heads,
            #    qkv_bias=True,
            #    qk_scale=None,
            #    drop=0.,
            #    attn_drop=0.,
            #    norm_layer=norm_layer,
            #    out_dim=clip_embed_dim)
            #fc_norm = build_norm_layer(clip_embed_dim, norm_layer, eps=1e-6)
            #if num_classes > 0:
            #    output = layers.Dense(clip_embed_dim, num_classes)
            #else:
            output = x


    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, output, name="InterImage")

    return model


def StemLayer(x,
              name,
              out_chans=96,
              act_layer='GELU',
              norm_layer='BN'):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    x = layers.Conv2D(filters=out_chans // 2,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      name=name +"_conv1")(x)
    x = build_norm_layer(norm_layer, name=name+"_norm1")(x)
    x = build_act_layer(act_layer, name=name+"_act")(x)
    x = layers.Conv2D(filters=out_chans,
                      kernel_size=3,
                      strides=2,
                      padding="same",
                      name=name+"_conv2")(x)
    x = build_norm_layer(norm_layer, name=name+"_norm2")(x)
    return x

#@tf.keras.utils.register_keras_serializable()
def InternImageBlock(x,
                     name,
                     channels,
                     depth,
                     groups,
                     return_wo_downsample=False,
                     downsample=True,
                     mlp_ratio=4.,
                     drop=0.,
                     drop_path=0.,
                     act_layer='GELU',
                     norm_layer='LN',
                     post_norm=False,
                     offset_scale=1.0,
                     layer_scale=None,
                     with_cp=False,
                     dw_kernel_size=None,  # for InternImage-H/G
                     post_norm_block_ids=None,  # for InternImage-H/G
                     res_post_norm=False,  # for InternImage-H/G
                     center_feature_scale=False):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """
    for i in range(depth):
        x = InternImageLayer(x,
                name=name+"_block" + str(i),
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size, # for InternImage-H/G
                res_post_norm=res_post_norm, # for InternImage-H/G
                center_feature_scale=center_feature_scale # for InternImage-H/G
        )
    if not post_norm or center_feature_scale:
        x = build_norm_layer('LN', name=name+"_norm")(x)

    if post_norm_block_ids is not None: # for InternImage-H/G
        for i, _ in enumerate(post_norm_block_ids):
            x = build_norm_layer('LN', name=name+"_post_norms"+str(i), eps=1e-6)(x)
    if return_wo_downsample:
        x_ = x
    if downsample:
        x = DownsampleLayer(x, name=name+"_downsample", channels=channels, norm_layer=norm_layer)
    if return_wo_downsample:
        return x, x_
    return x


def InternImageLayer(x,
                     name,
                     channels,
                     groups,
                     mlp_ratio=4.,
                     drop=0.,
                     drop_path=0.,
                     act_layer='GELU',
                     norm_layer='LN',
                     post_norm=False,
                     layer_scale=None,
                     offset_scale=1.0,
                     with_cp=False,
                     dw_kernel_size=None,  # for InternImage-H/G
                     res_post_norm=False,  # for InternImage-H/G
                     center_feature_scale=False
                     ):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """
    if not layer_scale:
        if post_norm:
            x_ = x
            x = DCNv3(name=name + "_dcn", channels=channels, kernel_size=3, stride=1,
                      pad=1, dilation=1, group=groups, offset_scale=offset_scale,
                      act_layer=act_layer, norm_layer=norm_layer,
                      dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                      center_feature_scale=center_feature_scale)(x)  # for InternImage-H/G
            x = build_norm_layer('LN', name=name + "_norm1")(x)
            x = DropPath(rate=drop_path, name=name + "_drop_path1")(x)
            x = layers.add([x,x_], name=name+"_add1")

            x_ = x
            x = MLPLayer(x, name=name + "_mlp", in_features=channels, hidden_features=int(channels * mlp_ratio),
                         act_layer=act_layer, drop=drop)
            x = build_norm_layer('LN', name=name + "_norm2")(x)
            x = DropPath(rate=drop_path, name=name + "_drop_path2")(x)
            x = layers.add([x,x_], name=name+"_add2")
        elif res_post_norm:  # for InternImage-H/G
            x_ = x
            x = build_norm_layer('LN', name=name + "_norm1")(x)
            x = DCNv3(name=name + "_dcn", channels=channels, kernel_size=3, stride=1,
                      pad=1, dilation=1, group=groups, offset_scale=offset_scale,
                      act_layer=act_layer, norm_layer=norm_layer,
                      dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                      center_feature_scale=center_feature_scale)(x)  # for InternImage-H/G
            x = build_norm_layer('LN', name=name + "_res_post_norm1")(x)
            x = DropPath(rate=drop_path, name=name + "_drop_path1")(x)
            x = layers.add([x,x_], name=name+"_add1")

            x_ = x
            x = build_norm_layer('LN', name=name + "_norm2")(x)
            x = MLPLayer(x, name=name + "_mlp", in_features=channels, hidden_features=int(channels * mlp_ratio),
                         act_layer=act_layer, drop=drop)
            x = build_norm_layer('LN', name=name + "_res_post_norm2")(x)
            x = DropPath(rate=drop_path, name=name + "_drop_path2")(x)
            x = layers.add([x,x_], name=name+"_add2")
        else:
            x_ = x
            x = build_norm_layer('LN', name=name + "_norm1")(x)
            x = DCNv3(name=name + "_dcn", channels=channels, kernel_size=3, stride=1,
                      pad=1, dilation=1, group=groups, offset_scale=offset_scale,
                      act_layer=act_layer, norm_layer=norm_layer,
                      dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                      center_feature_scale=center_feature_scale)(x)  # for InternImage-H/G
            x = DropPath(rate=drop_path, name=name + "_drop_path1")(x)
            x = layers.add([x,x_], name=name+"_add1")

            x_ = x
            x = build_norm_layer('LN', name=name + "_norm2")(x)
            x = MLPLayer(x, name=name + "_mlp", in_features=channels, hidden_features=int(channels * mlp_ratio),
                         act_layer=act_layer, drop=drop)
            x = DropPath(rate=drop_path, name=name + "_drop_path2")(x)
            x = layers.add([x,x_], name=name+"_add2")
        return x
    gamma1 = tf.Variable(layer_scale * tf.ones(channels))
    gamma2 = tf.Variable(layer_scale * tf.ones(channels))
    if post_norm:
        x_ = x
        x = DCNv3(name=name + "_dcn", channels=channels, kernel_size=3, stride=1,
                  pad=1, dilation=1, group=groups, offset_scale=offset_scale,
                  act_layer=act_layer, norm_layer=norm_layer,
                  dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                  center_feature_scale=center_feature_scale)(x)  # for InternImage-H/G
        x = build_norm_layer('LN', name=name + "_norm1")(x)
        x = DropPath(rate=drop_path, name=name + "_drop_path1")(x * gamma1)
        x = layers.add([x,x_], name=name+"_add1")

        x_ = x
        x = MLPLayer(x, name=name + "_mlp", in_features=channels, hidden_features=int(channels * mlp_ratio),
                     act_layer=act_layer, drop=drop)
        x = build_norm_layer('LN', name=name + "_norm2")(x)
        x = DropPath(rate=drop_path, name=name + "_drop_path2")(x * gamma2)
        x = layers.add([x,x_], name=name+"_add2")
    else:
        x_ = x
        x = build_norm_layer('LN', name=name + "_norm1")(x)
        x = DCNv3(name=name + "_dcn", channels=channels, kernel_size=3, stride=1,
                  pad=1, dilation=1, group=groups, offset_scale=offset_scale,
                  act_layer=act_layer, norm_layer=norm_layer,
                  dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                  center_feature_scale=center_feature_scale)(x)  # for InternImage-H/G
        x = DropPath(rate=drop_path, name=name + "_drop_path1")(x * gamma1)
        x = layers.add([x,x_], name=name+"_add1")

        x_ = x
        x = build_norm_layer('LN', name=name + "_norm2")(x)
        x = MLPLayer(x, name=name + "_mlp", in_features=channels, hidden_features=int(channels * mlp_ratio),
                     act_layer=act_layer, drop=drop)
        x = DropPath(rate=drop_path, name=name + "_drop_path2")(x * gamma2)
        x = layers.add([x,x_], name=name+"_add2")
    return x


def MLPLayer(x,
             name,
             in_features,
             hidden_features=None,
             out_features=None,
             act_layer='GELU',
             drop=0.
             ):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    x = layers.Dense(units=hidden_features, name=name+"_fc1")(x)
    x = build_act_layer(act_layer, name=name+"_act")(x)
    x = layers.Dropout(drop, name=name + "_drop1")(x)
    x = layers.Dense(units=out_features, name=name+"_fc2")(x)
    x = layers.Dropout(drop, name=name+"_drop2")(x)
    return x

class DCNv3(tf.keras.layers.Layer):
    def __init__(
            self,
            name="_dcn",
            channels=64,
            kernel_size=3,
            dw_kernel_size=None,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            act_layer='GELU',
            norm_layer='LN',
            center_feature_scale=False):
        """
        DCNv3 Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__(dynamic=True)
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.P = kernel_size * kernel_size

        self.dw_conv = [
            layers.Conv2D(
                filters=channels,
                kernel_size=dw_kernel_size,
                strides=1,
                padding="same",
                groups=channels,
                name=name+"_dw_conv"),
            build_norm_layer(norm_layer, name=name+"_dw_norm"),
            build_act_layer(act_layer, name=name+"_dw_act")]
        self.offset = layers.Dense(units=self.group * self.P * 2, name=name+"_offset")
        self.mask = layers.Dense(units=self.group * self.P, name=name+"_mask")
        self.input_proj = layers.Dense(units=self.channels, name=name+"_input_proj")
        self.output_proj = layers.Dense(units=self.channels, name=name+"_output_proj")

        if center_feature_scale:
            self.center_feature_scale_module = layers.Dense(units=self.group, name=name+"_cfcm")

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.channels, )

    def call(self, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x

        x1=input
        for layer in self.dw_conv:
            x1 = layer(x1)
        offset = self.offset(x1)
        mask = tf.reshape(self.mask(x1), [-1, H, W, self.group, self.P])
        mask = tf.reshape(backend.softmax(mask, -1), [-1, H, W, self.group * self.P])

        x = dcnv3_core(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        if self.center_feature_scale:
            center_feature_scale = backend.sigmoid(self.center_feature_scale_module(x1), -1)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = tf.reshape(tf.repeat(tf.expand_dims(center_feature_scale[..., None],-1), self.channels // self.group, -1), [N, H, W, -1])
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


def DownsampleLayer(x, name, channels, norm_layer='LN'):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    x = layers.Conv2D(filters=2 * channels,
                          kernel_size=3,
                          strides=(2, 2),
                          padding="same",
                          use_bias=False,
                          name=name+"_conv")(x)
    x = build_norm_layer(norm_layer, name=name+"_norm")(x)
    return x


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


def dcnv3_core(
        input, offset, mask, kernel_h,
        kernel_w, stride_h, stride_w, pad_h,
        pad_w, dilation_h, dilation_w, group,
        group_channels, offset_scale):
    # for debug and test only,
    # need to use cuda version instead
    input = tf.pad(input, [[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape
    P_ = kernel_h * kernel_w

    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    #spatial_norm = tf.repeat(tf.reshape(tf.constant([W_in, H_in], dtype=tf.float32), [1, 1, 1, 2]), group*kernel_h*kernel_w, 3)

    # N_, H_out, W_out, group*P_*2
    sampling_locations = tf.reshape(tf.repeat((ref + grid * offset_scale), N_, 0), [N_, H_out, W_out, group*P_*2]) + offset * offset_scale #/ spatial_norm

    #sampling_grids = 2 * sampling_locations - 1
    sampling_grids = tf.reshape(sampling_locations, [N_, H_out*W_out, group, P_, 2])
    # N_, H_in, W_in, group*group_channels -> N_, H_in*W_in, group*group_channels -> N_, group*group_channels, H_in*W_in -> N_*group, group_channels, H_in, W_in
    input_ = tf.reshape(tf.transpose(tf.reshape(input,[N_, H_in*W_in, group, group_channels]), [0, 2, 1, 3]), [N_*group, H_in, W_in, group_channels])
    # N_, H_out, W_out, group*P_*2 -> N_, H_out*W_out, group, P_, 2 -> N_, group, H_out*W_out, P_, 2 -> N_*group, H_out*W_out, P_, 2
    sampling_grid_ = tf.reshape(tf.transpose(sampling_grids, [0, 2, 1, 3, 4]), [N_*group, H_out*W_out*P_, 2])
    # N_*group, group_channels, H_out*W_out, P_

    sampling_grid_ = tf.cast(sampling_grid_, tf.int32)
    # N_, H_out, W_out, group*P_
    sampling_input_ = tf.reshape(tf.gather_nd(input_, sampling_grid_, batch_dims=1), [N_*group, H_out, W_out, P_, group_channels])

    # (N_, H_out, W_out, group*P_) -> N_, H_out*W_out, group, P_ -> (N_, group, H_out*W_out, P_) -> (N_*group, 1, H_out*W_out, P_)
    #mask = tf.reshape(tf.transpose(tf.reshape(mask, [N_, H_out*W_out, group, P_]), [0, 2, 1, 3]), [N_*group, 1, H_out*W_out, P_])
    mask = tf.reshape(tf.transpose(tf.reshape(mask, [N_, H_out, W_out, group, P_]), [0, 3, 1, 2, 4]), [N_ * group, H_out, W_out, P_, 1])

    output = tf.reshape(tf.reduce_sum((sampling_input_ * mask), -2), [N_, group, H_out, W_out, group_channels])

    return tf.reshape(tf.transpose(output, [0, 2, 3, 1, 4]), [N_, H_out, W_out, -1])

def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = tf.meshgrid(
        tf.linspace(
            # pad_h + 0.5,
            # H_ - pad_h - 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            ),
        tf.linspace(
            # pad_w + 0.5,
            # W_ - pad_w - 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            ))
    ref_y = tf.reshape(ref_y, [-1])[None] #/ H_
    ref_x = tf.reshape(ref_x, [-1])[None] #/ W_

    ref = tf.reshape(tf.stack((ref_y, ref_x), -1), [1, H_out, W_out, 1, 2])

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = tf.meshgrid(
        tf.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            ),
        tf.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            ))

    #points_list.extend([x / W_, y / H_])
    points_list.extend([y, x])
    grid = tf.transpose(tf.repeat(tf.reshape(tf.stack(points_list, -1), [-1, 1, 2]), group, 1), perm=[1, 0, 2])
    grid = tf.reshape(grid, [1, 1, 1, group * kernel_h * kernel_w, 2])

    return tf.cast(grid, tf.float32)

#class DropPath(tf.keras.__internal__.layers.BaseRandomLayer):
class DropPath(tf.keras.layers.Layer):
    """
    Implements the DropPath layer. DropPath randomly drops samples during training
     with a probability of `rate`. Note that this layer drops individual samples
    within a batch and not the entire batch. DropPath randomly drops some of the
    individual samples from a batch, whereas StachasticDepth randomly drops the
    entire batch.

    References:
        - [FractalNet](https://arxiv.org/abs/1605.07648v4).
        - [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135)

    Args:
        rate: float, the probability of the residual branch being dropped.
        seed: (Optional) Integer. Used to create a random seed.

    Usage:
    `DropPath` can be used in any network as follows:
    ```python

    # (...)
    input = tf.ones((1, 3, 3, 1), dtype=tf.float32)
    residual = tf.keras.layers.Conv2D(1, 1)(input)
    output = keras_cv.layers.DropPath()(input)
    # (...)
    ```
    """

    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__()
        self.rate = rate
        self.seed = seed

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            keep_prob = 1 - self.rate
            drop_map_shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
            drop_map = tf.keras.backend.random_bernoulli(
                drop_map_shape, p=keep_prob, seed=self.seed
            )
            x = x / keep_prob
            x = x * drop_map
            return x

    def get_config(self):
        config = {"rate": self.rate, "seed": self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    print("Hello world!")
    model = InternImage()
    dummy = tf.zeros((1,128,64,3))
    print("Build with dummy Tensor of shape: ")
    print(dummy.shape)
    #model.build(dummy.shape)
    y = model(dummy)
    model.summary()
