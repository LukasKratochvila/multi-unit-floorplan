import logging
import os
import time
import argparse

import cv2
import matplotlib.image as mpimg

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image

from pdf2image import convert_from_path

from utils import ind2rgb, Config
from correct_result import convert_values
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

logging.disable(logging.WARNING)

Image.MAX_IMAGE_PIXELS = None
# Define max image width - need approximately 5GB memory
w_n = 1000

def pad_to(x, stride):
    h, w = x.shape[:2]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w

    new_h = max(new_h, new_w)
    new_w = max(new_h, new_w)
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = tf.constant([[lh, uh], [lw, uw], [0, 0]])

    # zero-padding by default.
    out = tf.pad(x, pads, "CONSTANT", 255)

    return out, pads


def unpad(x, pad):
    return x[pad[0, 0]:-pad[0, 1], pad[1, 0]:-pad[1, 1]]


def predict(cfg):

    unet_model = tf.keras.models.load_model(cfg.train_cfg.log_dir, compile=False)

    images = []
    for img in cfg.predict_cfg.images:
        if img.split(".")[-1] == "pdf":
            pages = convert_from_path(img, 300)
            for page in pages:
                # resize image because max image size
                w, h = page.size
                ratio = h/w
                page = page.resize((w_n, int(w_n*ratio)))
                images.append(np.array(page))
        else:
            images.append(cv2.imread(img))

    for i, image_org in enumerate(tqdm(images, 'Processing images')):

        image_org = np.clip(image_org, 0, 255)
        image_org, pads = pad_to(image_org, cfg.predict_cfg.pad)

        image = tf.convert_to_tensor(image_org, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)

        shp = image.shape
        if cfg.train_cfg.normalize:
            image = tf.reshape(image, [shp[0], shp[1], 3]) / 255 
        else:
            image = tf.reshape(image, [shp[0], shp[1], 3])

        image_batch = tf.expand_dims(image, axis=0)
        
        tic = time.time()
        prediction = unet_model.predict_on_batch(image_batch)
        toc = time.time()
        
        cfg.predict_cfg.raw_predict_time = toc - tic
        
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        result = prediction[0].argmax(axis=-1).astype(np.uint8)

        timestr = time.strftime("%Y%m%d-%H%M%S")

        # Unpad result
        if np.any(pads.numpy()):
            result = unpad(result, pads)

        # Fix model output
        classes = ['background', 'walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
        model_classes = ['background'] + cfg.train_cfg.classes
        result = convert_values(result, model_classes, classes)

        # Save output
        cfg.predict_cfg.save_paths.append(
            '_'.join([os.path.basename(cfg.predict_cfg.images[i]).split('.')[0], cfg.train_cfg.exp_name, timestr]) + '.png')
        mpimg.imsave(cfg.predict_cfg.output_dir + "/result_" + cfg.predict_cfg.save_paths[-1], ind2rgb(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict trained model on image or image folder.')
    parser.add_argument('config', help='Config file path (could be multiple)', nargs='+')
    parser.add_argument('images', default='', help='Image or folder to process')
    parser.add_argument('--output_dir', default='results/', help='Folder to save output')

    args = parser.parse_args()
    configs = args.config
    images = args.images
    if os.path.isdir(images):
        images = [args.images + file for file in os.listdir(images)]
    else:
        images = [images]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for config in configs:
        print('Processing config: ', config)
        # load train_cfg
        cfg = Config.fromfile(config)
        # load dataset info
        cfg.predict_cfg = dict()
        # prepare variables
        cfg.predict_cfg.images = images
        cfg.predict_cfg.save_paths = []
        cfg.predict_cfg.output_dir = args.output_dir
        nup = cfg.train_cfg.get('n_up_sample_block', 5)
        if nup is None:
            nup = 5
        cfg.predict_cfg.pad = tf.math.pow(2, nup)
        tic = time.time()
        predict(cfg)
        toc = time.time()
        print('total predict time = {} minutes'.format((toc - tic) / 60))

        cfg.predict_cfg.whole_predict_time = toc - tic
        cfg.dump(os.path.join(cfg.train_cfg.log_dir, 'predict_cfg.py'))
    print('============== End ==============')
