"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import sklearn
from scipy import interpolate
import datetime

max_threshold = 0
min_threshold = 4

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    global max_threshold
    global min_threshold

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # print('train_set', train_set)
        # print('test_set', test_set)
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            # print(_embed_train.shape)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            # print(embed1.shape, embed2.shape)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        # print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set], is_show_log=True, fold_idx=fold_idx)

        if max_threshold < thresholds[best_threshold_index]:
            max_threshold = thresholds[best_threshold_index]
        if min_threshold > thresholds[best_threshold_index]:
            min_threshold = thresholds[best_threshold_index]
    print('thresholds max: {} <=> min: {}'.format(max_threshold, min_threshold))

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame, is_show_log=False, fold_idx="none"):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size

    # if is_show_log:
    # print("")
    # print("----", threshold, "fold_idx:", fold_idx, "-------------------------------------------------")
    # for i in range(len(predict_issame)):
    # 距離分布出力
    # print(fold_idx * len(predict_issame) + i, dist[i], actual_issame[i])

    # # ミスのケース表示
    # if predict_issame[i] != actual_issame[i]:
    #     print("diff", (i + fold_idx * 600) * 2, "vs", (i + fold_idx * 600) * 2 + 1, "label:", i + fold_idx * 600, "actual:", actual_issame[i], "predict:", predict_issame[i], "distance:", dist[i])

    # # 正解で遠めを検出
    # if predict_issame[i] == actual_issame[i] and dist[i] > 1.15 and dist[i] < 1.3:
    #     print("same far", (i + fold_idx * 600) * 2, "vs", (i + fold_idx * 600) * 2 + 1, "label:", i + fold_idx * 600, "actual:", actual_issame[i], "predict:", predict_issame[i], "distance:", dist[i])

    # # 正解で近めを検出
    # if predict_issame[i] == actual_issame[i] and ((dist[i] > 0.3 and dist[i] < 0.35) or (dist[i] > 2.3 and dist[1] < 2.35)):
    #     print("same near", (i + fold_idx * 600) * 2, "vs", (i + fold_idx * 600) * 2 + 1, "label:", i + fold_idx * 600, "actual:", actual_issame[i], "predict:", predict_issame[i], "distance:", dist[i])

    # print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)
    # accuracy = (tp + tn) / (tp + fp + tn + fn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # specificity = tn / (fp + tn)
    # f_measure = (2 * recall * precision) / (recall + precision)
    # print("accuracy:", accuracy)
    # print("precision:", precision)
    # print("recall:", recall)
    # print("specificity:", specificity)
    # print("f-measure:", f_measure)

    return tpr, fpr, acc

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    '''
    Copy from [insightface](https://github.com/deepinsight/insightface)
    :param thresholds:
    :param embeddings1:
    :param embeddings2:
    :param actual_issame:
    :param far_target:
    :param nrof_folds:
    :return:
    '''
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def data_iter(datasets, batch_size):
    data_num = datasets.shape[0]
    for i in range(0, data_num, batch_size):
        yield datasets[i:min(i+batch_size, data_num), ...]


def test(data_set, sess, embedding_tensor, batch_size, label_shape=None, feed_dict=None, input_placeholder=None):
    '''
    referenc official implementation [insightface](https://github.com/deepinsight/insightface)
    :param data_set:
    :param sess:
    :param embedding_tensor:
    :param batch_size:
    :param label_shape:
    :param feed_dict:
    :param input_placeholder:
    :return:
    '''
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        datas = data_list[i]
        embeddings = None
        feed_dict.setdefault(input_placeholder, None)
        for idx, data in enumerate(data_iter(datas, batch_size)):
            data_tmp = data.copy()    # fix issues #4
            data_tmp -= 127.5
            data_tmp *= 0.0078125
            feed_dict[input_placeholder] = data_tmp
            time0 = datetime.datetime.now()
            _embeddings = sess.run(embedding_tensor, feed_dict)
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((datas.shape[0], _embeddings.shape[1]))
            try:
                embeddings[idx*batch_size:min((idx+1)*batch_size, datas.shape[0]), ...] = _embeddings
            except ValueError:
                print('idx*batch_size value is %d min((idx+1)*batch_size, datas.shape[0]) %d, batch_size %d, data.shape[0] %d' %
                      (idx*batch_size, min((idx+1)*batch_size, datas.shape[0]), batch_size, datas.shape[0]))
                print('embedding shape is ', _embeddings.shape)
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            # print(_em.shape, _norm)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=10)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list


def ver_test(ver_list, ver_name_list, nbatch, sess, embedding_tensor, batch_size, feed_dict, input_placeholder):
    results = []
    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = test(data_set=ver_list[i], sess=sess, embedding_tensor=embedding_tensor,
                                                              batch_size=batch_size, feed_dict=feed_dict,
                                                              input_placeholder=input_placeholder)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
    return results
