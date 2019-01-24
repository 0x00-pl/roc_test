import math
import os
import scipy
from multiprocessing import Pool
from scipy import spatial
import dtw
import numpy as np
import pydub
import python_speech_features
from pydub import effects
import matplotlib.pyplot as plt
import webrtcvad
import itertools


def load_file(filename, file_format, frame_rate=16000):
    sound = pydub.AudioSegment.from_file(filename, file_format)
    sound = sound.set_frame_rate(frame_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound = sound.remove_dc_offset()
    sound = effects.normalize(sound)
    signal = np.array(sound.get_array_of_samples())

    # preemph = 0.97
    # signal = python_speech_features.sigproc.preemphasis(signal, preemph)

    vader = webrtcvad.Vad()
    vader.set_mode(1)
    frames = python_speech_features.sigproc.framesig(signal, 320, 160)
    frames = np.array([i for i in frames if vader.is_speech(i.astype('int16').tobytes(), 16000)])
    signal = frames.flatten()

    # ret = python_speech_features.sigproc.powspec(frames, 320)
    # ret = python_speech_features.fbank(signal, winlen=0.02, winstep=0.02, winfunc=lambda x: np.hamming(x))[0]
    ret = python_speech_features.mfcc(signal, numcep=13, nfilt=26, winlen=0.02, winstep=0.02, lowfreq=100, winfunc=lambda x: np.hamming(x))
    ret = ret - np.mean(ret, axis=0)
    ret = ret / np.var(ret, axis=0)

    ret_delta = python_speech_features.delta(ret, 1)
    # ret_delta = ret_delta - np.mean(ret_delta, axis=0)
    # ret_delta = ret_delta / np.var(ret_delta, axis=0)
    ret_delta2 = python_speech_features.delta(ret_delta, 1)
    # ret_delta2 = ret_delta2 - np.mean(ret_delta2, axis=0)
    # ret_delta2 = ret_delta2 / np.var(ret_delta2, axis=0)

    ret_acc = np.array(list(itertools.accumulate(ret, (lambda prev, cur: prev/2+cur))))
    ret_acc = ret_acc - np.mean(ret_acc, axis=0)
    ret_acc = ret_acc / np.var(ret_acc, axis=0)

    # # ret[ret <= 1e-30] = 1e-30
    return np.concatenate((ret, ret_delta, ret_delta2, ret_acc), axis=1)
    # ret = np.add.accumulate(ret)
    # ret_delta = np.add.accumulate(ret_delta)
    # ret_delta2 = np.add.accumulate(ret_delta2)
    # return np.concatenate((ret, ret_delta, ret_delta2), axis=1)


def read_all(root_path='kanzhitongxue'):
    ret = []
    for prefix_path, dirs, files in os.walk(root_path):
        for filename in files:
            file_path = os.path.join(prefix_path, filename)
            feature_list = load_file(file_path, 'wav')
            ret.append([feature_list, filename])

    return ret


l2_norm = lambda x, y: np.linalg.norm((x - y), ord=2)
l2_norm_fast = lambda x, y: math.sqrt(np.product((x - y) ** 2))
cos_dist = 'cosine'
euclidean_dist = 'euclidean'


def get_dtw_distance(feature_list1, feature_list2, a_label, b_label):
    d, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(feature_list1, feature_list2, dist='cosine', warp=1)

    # plt.imshow(cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.xlabel('label: ' + str(a_label))
    # plt.ylabel('label: ' + str(b_label))
    # plt.title(str(a_label[0] == b_label[0]) + ' dist:' + str(d))
    # print(a_label, b_label)
    # plt.show()
    return d


def cal_distance(label_feature_list, get_distance):
    ret = []
    for a in range(len(label_feature_list) - 1):
        for b in range(a + 1, len(label_feature_list)):
            a_label, a_feature = label_feature_list[a]
            b_label, b_feature = label_feature_list[b]
            distance = get_distance(a_feature, b_feature, a_label, b_label)
            ret.append([distance, a_label, b_label])

    return ret


def cal_distance_multiprocessing_step(args):
    feature1, feature2, label1, label2 = args
    distance = get_dtw_distance(feature1, feature2, label1, label2)
    return [distance, label1, label2]


def cal_distance_multiprocessing(label_feature_list):
    args_list = []
    for a in range(len(label_feature_list)):
        for b in range(a, len(label_feature_list)):
            a_label, a_feature = label_feature_list[a]
            b_label, b_feature = label_feature_list[b]
            args_list.append([a_feature, b_feature, a_label, b_label])

    with Pool(processes=8) as pool:
        return pool.map(cal_distance_multiprocessing_step, args_list, chunksize=1000)

    # ret = []
    # for args in args_list:
    #     ret.append(cal_distance_multiprocessing_step(args))
    # return ret


def cal_trueneg_falsepos_step(distance_label_list, threshold):
    trueneg_sum, falsepos_sum = 0, 0
    for distance, label1, label2 in distance_label_list:
        y = distance < threshold
        if label1[0] == label2[0] and not y:
            trueneg_sum = trueneg_sum + 1
        elif label1[0] != label2[0] and y:
            falsepos_sum = falsepos_sum + 1
    return trueneg_sum, falsepos_sum, threshold


def cal_trueneg_falsepos(distance_label_list, threshold_list):
    return [cal_trueneg_falsepos_step(distance_label_list, threshold) for threshold in threshold_list]


def create_range(minv, maxv, countv=100):
    props = np.array(range(countv)) / countv
    return minv + props * (maxv - minv)


def print_roc(roc_list, name='roc_cos'):
    print(name, '=', end='[')
    for trueneg, falsepos, threahold in roc_list:
        print(trueneg, falsepos, threahold, end=';')
    print('];')


def main():
    distance_type = 'quanzidong_cos'
    if os.path.exists(distance_type + '_distance_label_list.npy') and False:
        distance_label_list = np.load(distance_type + '_distance_label_list.npy')
    else:
        pos_features_name = read_all('kanzhitongxue/pos')
        neg_features_name = read_all('kanzhitongxue/other_text')
        label_feature_list = [([1, name], fa) for fa, name in pos_features_name] + [([0, name], fa) for fa, name in neg_features_name]
        distance_label_list = cal_distance_multiprocessing(label_feature_list)
        np.save(distance_type + '_distance_label_list.npy', distance_label_list)

    dist_min = min(distance_label_list, key=(lambda x: x[0]))[0]
    dist_max = max(distance_label_list, key=(lambda x: x[0]))[0]
    # dist_min = 0
    # dist_max = 36591210330589

    threshold_range = create_range(dist_min, dist_max, 1000)
    roc_list = cal_trueneg_falsepos(distance_label_list, threshold_range)
    print_roc(roc_list, 'roc_dtw')


if __name__ == '__main__':
    main()
