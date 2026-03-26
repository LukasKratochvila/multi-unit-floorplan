import time

import tensorflow as tf

from datasets import floorplans
from training.aaf_layers import ignores_from_label, edges_from_label, eightcorner_activation
from training.loss_functions import asymmetric_focal_tversky_loss, asymmetric_focal_loss
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import ind2rgb, door_map


def ul_test(y_true, y_pred, weight=0.5, delta=0.6, gamma=0.5, num_classes=1):
    y_true = y_true[:, :, :, :num_classes]
    asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true, y_pred)
    asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true, y_pred)

    if weight is not None:
        return (weight * asymmetric_ftl) + ((1 - weight) * asymmetric_fl)


def aaf_test(y_true, y_pred,
             size,
             num_classes,
             w_edge=0.5,
             w_not_edge=0.5,
             kld_margin=3.0):
    y_true = y_true[:, :, :, :num_classes]

    # Downsample to save memory
    new_d = y_true.get_shape()[1] // 2
    # y_true = tf.image.resize(y_true, [new_d, new_d], method='nearest')
    # y_pred = tf.image.resize(y_pred, [new_d, new_d], method='nearest')

    # Compute ignore map (e.g, label of 255 and their paired pixels).
    labels = tf.keras.backend.argmax(y_true, axis=-1)  # NxHxW TODO maybe need squeeze still check source code
    # labels = tf.squeeze(tf.keras.backend.argmax(y_true, axis=-1), axis=-1)  # NxHxW TODO maybe need squeeze still check source code
    ignore = ignores_from_label(labels, num_classes, size)  # NxHxWx8
    not_ignore = tf.logical_not(ignore)
    not_ignore = tf.expand_dims(not_ignore, axis=3)  # NxHxWx1x8

    # Compute edge map.
    edge = edges_from_label(y_true, size, 0)  # NxHxWxCx8

    # Remove ignored pixels from the edge/non-edge.
    edge = tf.logical_and(edge, not_ignore)
    not_edge = tf.logical_and(tf.logical_not(edge), not_ignore)

    edge_indices = tf.where(tf.reshape(edge, [-1]))
    not_edge_indices = tf.where(tf.reshape(not_edge, [-1]))

    # Extract eight corner from the center in a patch as paired pixels.
    probs_paired = eightcorner_activation(y_pred, size)  # NxHxWxCx8
    probs = tf.expand_dims(y_pred, axis=-1)  # NxHxWxCx1
    bot_epsilon = tf.constant(1e-4, name='bot_epsilon')
    top_epsilon = tf.constant(1.0, name='top_epsilon')

    neg_probs = tf.clip_by_value(
        1 - probs, bot_epsilon, top_epsilon)
    neg_probs_paired = tf.clip_by_value(
        1 - probs_paired, bot_epsilon, top_epsilon)
    probs = tf.clip_by_value(
        probs, bot_epsilon, top_epsilon)
    probs_paired = tf.clip_by_value(
        probs_paired, bot_epsilon, top_epsilon)

    # Compute KL-Divergence.
    kldiv = probs_paired * tf.math.log(probs_paired / probs)
    kldiv += neg_probs_paired * tf.math.log(neg_probs_paired / neg_probs)
    edge_loss = tf.maximum(0.0, kld_margin - kldiv)
    not_edge_loss = kldiv

    # Impose weights on edge/non-edge losses.
    one_hot_lab = tf.expand_dims(y_true, axis=-1)
    # w_edge_sum = tf.reduce_sum((1 - w_edge_norm[..., k]) * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
    w_edge_sum = tf.reduce_sum(w_edge * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
    # w_edge = tf.math.reduce_sum(1 * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
    # w_not_edge_sum = tf.reduce_sum((1 - w_not_edge_norm[..., k]) * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
    w_not_edge_sum = tf.reduce_sum(w_not_edge * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1
    # w_not_edge = tf.math.reduce_sum(1 * one_hot_lab, axis=3, keepdims=True)  # NxHxWx1x1

    edge_loss *= w_edge_sum
    not_edge_loss *= w_not_edge_sum

    not_edge_loss = tf.reshape(not_edge_loss, [-1])
    not_edge_loss = tf.gather(not_edge_loss, not_edge_indices)
    edge_loss = tf.reshape(edge_loss, [-1])
    edge_loss = tf.gather(edge_loss, edge_indices)

    loss = 0.0
    scale = 0.75
    # print(tf.reduce_mean(edge_loss), tf.reduce_mean(not_edge_loss))
    if tf.greater(tf.size(edge_loss), 0):
        loss += 0.5 * 1 / scale * tf.reduce_mean(edge_loss)
    if tf.greater(tf.size(not_edge_loss), 0):
        loss += 20 * scale * tf.reduce_mean(not_edge_loss)
    return loss / num_classes


def heatmap_regression_loss(num_classes, opening_inds):
    def loss_function(y_true, y_pred):
        loss = 0
        for i, ind in enumerate(opening_inds):
            opening_pred = y_pred[:, :, :, ind]
            heatmap_true = y_true[:, :, :, num_classes + i]  # Use heatmap offset
            # loss += tf.reduce_sum(diff) / tf.math.count_nonzero(diff > 1e-3, dtype=tf.float32)
            # loss += tf.reduce_mean(tf.square(tf.abs(heatmap_true - opening_true) + 1) - 1)
            # loss += tf.reduce_mean(tf.abs(heatmap_true - opening_true))
            # loss = tf.reduce_mean(diff)
            loss += tf.reduce_mean(tf.square(tf.abs(heatmap_true - opening_pred) + 1) - 1)
        return loss

    return loss_function


if __name__ == '__main__':

    # for i in range(3):
    #     img = cv2.imread('annotations/hdd/multi_plans/image' + str(i) + '/image.jpg')
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     mask = cv2.imread('annotations/hdd/multi_plans/image' + str(i) + '/mask.png', 0)
    #     # mask[mask == 0] = img
    #     plt.figure(dpi=400)
    #     plt.imshow(img)
    #     plt.imshow(ind2rgba(mask), alpha=0.75)
    #     plt.axis('off')
    #     plt.savefig('mufp' + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    #     plt.show()

    # classes = ['walls', 'openings']
    # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
    classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
    classes = ['bg'] + classes
    datasets = [
        'multi_plans_augment'
        # 'cubicasa5k_augment',
        # 'r3d_augment'
    ]
    normalize = False

    # loss_func = heatmap_regression_loss(len(classes), [2])
    loss_func = heatmap_regression_loss(len(classes), [4,5,6])

    gen_data = False
    if gen_data:
        test_dataset = floorplans.load_test_data(classes, datasets, normalize=normalize)
        for (image, label) in test_dataset.take(1).batch(1):
            unet_model = tf.keras.models.load_model(
                # 'models/zeng_zeng_multi_final_vgg16_32,64,128,256,512_multi_plans_augment_20220728-135338',
                'models/unet3p_2samples_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-180530',
                compile=False)
            p1 = unet_model.predict_on_batch(image)

            unet_model = tf.keras.models.load_model(
                'models/unet3p_2samples_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-153551',
                compile=False)
            # 'models/unet3p_EfficientNetB2_64,128,256,512,1024_r3d_augment_20220721-223128', compile=False)
            p2 = unet_model.predict_on_batch(image)

            np.save('p1', p1)
            np.save('p2', p2)
            np.save('label', label)

    label = tf.convert_to_tensor(np.load('label.npy'))
    p1 = tf.convert_to_tensor(np.load('p1.npy'))
    p2 = tf.convert_to_tensor(np.load('p2.npy'))

    plot = False
    if plot:
        plt.imshow(ind2rgb(label[0, :, :, :len(classes)].numpy().argmax(axis=-1)))
        plt.show()
        plt.imshow(label[0, :, :, len(classes)].numpy())
        plt.show()
        plt.imshow(ind2rgb(p1[0].numpy().argmax(axis=-1), color_map=door_map))
        plt.show()
        plt.imshow(ind2rgb(p2[0].numpy().argmax(axis=-1), color_map=door_map))
        plt.show()

    # print(loss_func(label, p1), loss_func(label, p2))
    # print(ul_test(label, p1, num_classes=len(classes)), ul_test(label, p2, num_classes=len(classes)))

    # print('====UF====')
    # tic = time.time()
    # for i in range(100):
    #     d = ul_test(label, p1, num_classes=len(classes))
    # toc = time.time()
    # avg_time = (toc-tic) / 100
    # print('uf: ' + str(avg_time))
    #
    # print('====Heatmap====')
    # tic = time.time()
    # for i in range(100):
    #     d = loss_func(label, p1)
    # toc = time.time()
    # avg_time = (toc - tic) / 100
    # print('heatmap: ' + str(avg_time))

    print('====AAF====')
    tic = time.time()
    for i in range(33):
        d = aaf_test(label, p1, num_classes=len(classes), size=2)
    toc = time.time()
    avg_time = (toc - tic) / 33
    print('aaf: ' + str(avg_time))
    # tic = time.time()
    # for i in range(20):
    #     d = aaf_test(label, p1, num_classes=len(classes), size=8)
    # toc = time.time()
    # print(toc-tic)

    # for s in range(1, 9):
    #     print(aaf_test(label, p1, num_classes=len(classes), size=s),
    #           aaf_test(label, p2, num_classes=len(classes), size=s))
