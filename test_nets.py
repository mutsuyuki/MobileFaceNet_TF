# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

import sys

sys.path.append("MobileFaceNet_TF")
from verification import evaluate

from common import load_validation_images_and_labels
from common import images_to_grayscale

from scipy.optimize import brentq
from scipy import interpolate
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import re
import os


def parse_arguments(argv):
    # default params

    # # 学習済みモデルの設定
    # convert2grayscale = False
    # model_path = "./weights/pretrained/"

    # # グレー画像学習の設定
    # convert2grayscale = True
    # model_path = "./weights/retrained_mono/"

    # カラー画像再学習の設定
    convert2grayscale = False
    model_path = "./weights/retrained_color_interm/"

    eval_data_path = "./dataset/faces_ms1m-refine-v2_112x112_converted/faces_emore/test"
    batch_size = 10 # default is 100

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default=model_path)
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.',
                        default=batch_size)
    # parser.add_argument('--eval_datasets', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_data_path',
                        default=eval_data_path,
                        help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--convert2grayscale', type=bool,
                        help='convert train and test image to grayscale', default=convert2grayscale)

    return parser.parse_args(argv)


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def main(args):
    with tf.Graph().as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # prepare validate datasets
            validate_data_list = []
            validate_data_name_list = []
            for dataset_type in args.eval_datasets:
                print(f"begin convert {dataset_type} in {args.eval_data_path}")

                # add to list
                data_set = load_validation_images_and_labels(args.eval_data_path, dataset_type)
                validate_data_list.append(data_set)
                validate_data_name_list.append(dataset_type)

            if args.convert2grayscale:
                print("------ grayscale mode -------")
                for i in range(len(validate_data_list)):
                    validate_data_list[i] = (images_to_grayscale(validate_data_list[i][0]), validate_data_list[i][1])

            # # ケーススタディ用画像書き出し
            # for db_index in range(len(validate_data_list)):
            #     data_sets, issame_list = validate_data_list[db_index]
            #     for i in range(len(data_sets)):
            #         # 同じ人を別人と間違えた
            #         if i == 24 or i == 47 or i == 62 or i == 68 or i == 69 or i == 98 or i == 112 or i == 150 or i == 154 or i == 187 or i == 655 or i == 677 or i == 715 or i == 831 or i == 841 or i == 1227 or i == 1228 or i == 1229 or i == 1233 or i == 1245 or i == 1269 or i == 1307 or i == 1342 or i == 1393 or i == 1395 or i == 1484 or i == 1804 or i == 1833 or i == 1843 or i == 1851 or i == 1853 or i == 1861 or i == 1878 or i == 1906 or i == 1918 or i == 1919 or i == 1931 or i == 1932 or i == 2006 or i == 2007 or i == 2009 or i == 2054 or i == 2418 or i == 2437 or i == 2438 or i == 2498 or i == 2504 or i == 2521 or i == 2550 or i == 2551 or i == 2609 or i == 2661 or i == 3017 or i == 3154 or i == 3155 or i == 3186 or i == 3226 or i == 3257 or i == 3600 or i == 3629 or i == 3734 or i == 3765 or i == 3885 or i == 3891 or i == 3892 or i == 4208 or i == 4214 or i == 4219 or i == 4220 or i == 4241 or i == 4244 or i == 4306 or i == 4324 or i == 4365 or i == 4370 or i == 4487 or i == 5083 or i == 5639 or i == 5693 or i == 5694 :
            #             image1 = (np.array(data_sets[i * 2]) / 0.0078125 + 127.5).astype(np.uint8)
            #             image2 = (np.array(data_sets[i * 2 + 1]) / 0.0078125 + 127.5).astype(np.uint8)
            #             result = np.concatenate([image1, np.ones(image1.shape, dtype=np.uint8) * 128, image2], axis=1)
            #             import cv2
            #             cv2.imwrite("FN_" + str(i) + ".png", result)
            #
            #         # 別人を同じ人と間違えた
            #         if i == 476 or i == 573 or i == 914 or i == 1000 or i == 1041 or i == 1096 or i == 1537 or i == 1607 or i == 1619 or i == 1659 or i == 1672 or i == 1794 or i == 2158 or i == 2332 or i == 2355 or i == 2809 or i == 2871 or i == 3359 or i == 3374 or i == 3496 or i == 3502 or i == 3940 or i == 3991 or i == 4118 or i == 4124 or i == 4506 or i == 4584 or i == 4620 or i == 4664 or i == 5156 or i == 5225 or i == 5716 or i == 5865 :
            #             image1 = (np.array(data_sets[i * 2]) / 0.0078125 + 127.5).astype(np.uint8)
            #             image2 = (np.array(data_sets[i * 2 + 1]) / 0.0078125 + 127.5).astype(np.uint8)
            #             result = np.concatenate([image1, np.ones(image1.shape, dtype=np.uint8) * 128, image2], axis=1)
            #             import cv2
            #             cv2.imwrite("FP_" + str(i) + ".png", result)
            # exit()

            # Load the model
            load_model(args.model)

            # Get input and output tensors, ignore phase_train_placeholder for it have default value.
            inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            embedding_size = embeddings.get_shape()[1]

            for db_index in range(len(validate_data_list)):
                # Run forward pass to calculate embeddings
                print('\nRunnning forward pass on {} images'.format(validate_data_name_list[db_index]))
                data_sets, issame_list = validate_data_list[db_index]
                nrof_batches = data_sets.shape[0] // args.test_batch_size
                emb_array = np.zeros((data_sets.shape[0], embedding_size))

                start_time = time.time()
                for index in range(nrof_batches):
                    start_index = index * args.test_batch_size
                    end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                    feed_dict = {inputs_placeholder: data_sets[start_index:end_index, ...]}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
                    print(end_index, "images processed.")
                duration = time.time() - start_time

                tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)
                print("total time %.3fs to evaluate %d images of %s" % (duration, data_sets.shape[0], validate_data_name_list[db_index]))
                print('Accuracy: %1.4f+-%1.4f' % (np.mean(accuracy), np.std(accuracy)))
                print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
                print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

                auc = metrics.auc(fpr, tpr)
                print('Area Under Curve (AUC): %1.5f' % auc)
                eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
                print('Equal Error Rate (EER): %1.5f' % eer)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))