import math
import os

import itertools

import numpy as np
import sklearn
import webrtcvad
from hmmlearn import hmm
import pydub
from matplotlib import pyplot
from matplotlib.pyplot import figure, scatter, show
from pydub import effects
import python_speech_features


def load_file(filename, file_format, frame_rate=16000):
    sound = pydub.AudioSegment.from_file(filename, file_format)
    sound = sound.set_frame_rate(frame_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound = sound.remove_dc_offset()
    sound = effects.normalize(sound)
    signal = np.array(sound.get_array_of_samples())

    vader = webrtcvad.Vad()
    vader.set_mode(1)
    frames = python_speech_features.sigproc.framesig(signal, 160, 160)
    frames = np.array([i for i in frames if vader.is_speech(i.astype('int16').tobytes(), 16000)])
    signal = frames.flatten()

    ret = python_speech_features.mfcc(signal, numcep=13, nfilt=26, winlen=0.025, winstep=0.01, lowfreq=100,
                                      winfunc=lambda x: np.hamming(x))

    # ret = ret - np.mean(ret.flatten())
    # ret = ret / np.var(ret.flatten())
    # ret = ret - np.mean(ret, axis=0)
    # ret = ret / np.var(ret, axis=0)

    ret_delta = python_speech_features.delta(ret, 1)
    ret_delta2 = python_speech_features.delta(ret_delta, 1)

    # ret = ret - np.mean(ret, axis=1)[:, np.newaxis]
    # ret = ret / np.var(ret, axis=1)[:, np.newaxis]
    # ret_acc = np.array(list(itertools.accumulate(ret, (lambda prev, cur: prev / 2 + cur))))
    # ret_acc = ret_acc - np.mean(ret_acc, axis=0)
    # ret_acc = ret_acc / np.var(ret_acc, axis=0)

    # return np.concatenate((ret, ret_delta, ret_delta2, ret_acc), axis=1)
    return np.concatenate((ret, ret_delta, ret_delta2), axis=1)


def read_all(root_path='kanzhitongxue'):
    ret = []
    for prefix_path, dirs, files in os.walk(root_path):
        for filename in files:
            file_path = os.path.join(prefix_path, filename)
            feature_list = load_file(file_path, 'wav')
            ret.append([feature_list, file_path])

    return ret


def print_roc(roc_list, name='roc_cos'):
    print(name, '=', end='[')
    for trueneg, falsepos, threahold in roc_list:
        print(trueneg, falsepos, threahold, end=';')
    print('];')


def train_GMMHMM(dataset, GMMHMM_Models=None):
    GMMHMM_Models = GMMHMM_Models or {}
    for label in dataset.keys():
        if GMMHMM_Models.get(label) == None:
            model = hmm.GaussianHMM(
                n_components=16, n_iter=1000, covariance_type='diag',
                params='tmc', init_params='mc',
                verbose=True
            )
            model.startprob_ = [1] + [0] * (model.n_components - 1)
            model.transmat_ = [
                [0.0] * i + [0.5, 0.5] + [0] * (model.n_components - i - 2)
                for i in range(model.n_components - 1)
            ] + [[0]*(model.n_components - 1) + [1]]
        else:
            model = GMMHMM_Models[label]

        trainData = dataset[label]
        length = np.zeros([len(trainData), ], dtype=np.int)
        for m in range(len(trainData)):
            length[m] = trainData[m][0].shape[0]
        trainData = np.vstack([i[0] for i in trainData])
        model.fit(trainData, lengths=length)  # get optimal parameters
        GMMHMM_Models[label] = model
    return GMMHMM_Models


def main():
    trainDataSet = {
        0: read_all('kanzhitongxue_train/u8k'),
        1: read_all('kanzhitongxue_train/pos')
    }
    testDataSet = {
        0: read_all('kanzhitongxue_test/u8k'),
        1: read_all('kanzhitongxue_test/pos')
    }

    models = None
    for i in range(1):
        models = train_GMMHMM(trainDataSet, models)

        trueneg, falsepos = 0, 0
        true_score_0_list, true_score_1_list = [], []
        false_score_0_list, false_score_1_list = [], []

        for feature in testDataSet[0]:
            score_0 = models[0].score(feature[0]) / len(feature[0])
            score_1 = models[1].score(feature[0]) / len(feature[0])
            if score_0 < -10000 or score_1 < -10000:
                print('bad false', feature[1], score_0, score_1)
                continue

            false_score_0_list.append(score_0)
            false_score_1_list.append(score_1)
            if score_1 > score_0:
                falsepos = falsepos + 1

        for feature in testDataSet[1]:
            score_0 = models[0].score(feature[0]) / len(feature[0])
            score_1 = models[1].score(feature[0]) / len(feature[0])
            if score_0 < -10000 or score_1 < -10000:
                print('bad true', feature[1], score_0, score_1)
                continue

            true_score_0_list.append(score_0)
            true_score_1_list.append(score_1)
            if score_0 > score_1:
                trueneg = trueneg + 1

        false_score_0_list, false_score_1_list = np.array(false_score_0_list), np.array(false_score_1_list)
        true_score_0_list, true_score_1_list = np.array(true_score_0_list), np.array(true_score_1_list)

        false_list = np.array([false_score_0_list, false_score_1_list]).transpose()
        true_list = np.array([true_score_0_list, true_score_1_list]).transpose()
        pca = sklearn.decomposition.PCA()
        pca.fit(np.vstack([false_list, true_list]))
        false_list = pca.transform(false_list)
        true_list = pca.transform(true_list)

        total_len = len(testDataSet[0]) + len(testDataSet[1])
        print('total:', total_len,
              'falsepos:', falsepos, falsepos / len(testDataSet[0]),
              'trueneg:', trueneg, trueneg / len(testDataSet[1]))

        pyplot.figure()
        pyplot.scatter(false_list[:, 0], false_list[:, 1], marker='x', s=5)
        pyplot.scatter(true_list[:, 0], true_list[:, 1], marker='o', s=5)
        pyplot.show()
        pyplot.figure()
        pyplot.scatter(false_score_0_list, false_score_1_list, marker='x', s=5)
        pyplot.scatter(true_score_0_list, true_score_1_list, marker='o', s=5)
        pyplot.plot([false_score_0_list.min(), false_score_0_list.max()],
                    [false_score_0_list.min(), false_score_0_list.max()])
        pyplot.plot([false_score_1_list.min(), false_score_1_list.max()],
                    [false_score_1_list.min(), false_score_1_list.max()])
        # for x,y,s in zip(false_score_0_list, false_score_1_list, np.array(testDataSet[0])[:, 1]):
        #     pyplot.text(x,y,s)
        pyplot.title('{} {}'.format(falsepos, trueneg))
        pyplot.show()


if __name__ == '__main__':
    main()
