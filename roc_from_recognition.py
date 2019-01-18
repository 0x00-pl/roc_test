import numpy as np


def read_record(file_name):
    cos_dist_list = []
    n2_dist_list = []
    for line in open(file_name, 'r'):
        name, data = line.split(':')
        cos_dist, n2_dist = data.split(',')
        cos_dist_list.append(float(cos_dist))
        n2_dist_list.append(float(n2_dist))
    return cos_dist_list, n2_dist_list


def read_all(idx_range, file_prefix='recognition/', file_suffix='.recognition.txt', idx_len=4):
    ret_cos_dist = []
    ret_n2_dist = []
    for idx in idx_range:
        file_idx = ('0' * idx_len + str(idx))[-idx_len:]
        filename = file_prefix + file_idx + file_suffix
        cos_dist, n2_dist = read_record(filename)
        ret_cos_dist.append(cos_dist)
        ret_n2_dist.append(n2_dist)

    return ret_cos_dist, ret_n2_dist


def cal_trueneg_falsepos(samples, label, threshold):
    res = np.array(samples)>threshold
    trueneg, falsepos = 0,0
    for item, idx in zip(res, range(len(res))):
        if idx == label and not item:
            trueneg = trueneg + 1
        elif idx != label and item:
            falsepos = falsepos + 1

    return trueneg, falsepos


def cal_threshold(samples_list, threshold_range):
    ret = []
    for threshold in threshold_range:
        trueneg_sum = 0
        falsepos_sum = 0
        for samples, idx in zip(samples_list, range(len(samples_list))):
            trueneg, falsepos = cal_trueneg_falsepos(samples, idx, threshold)
            trueneg_sum = trueneg_sum + trueneg
            falsepos_sum = falsepos_sum + falsepos

        ret.append((trueneg_sum, falsepos_sum, threshold))
    return ret


def create_range(minv, maxv, countv=100):
    l = np.array(range(countv))/countv
    return minv + l * (maxv-minv)


def main():
    cos_dist, n2_dist = read_all(range(1,101))
    cos_dist = np.array(cos_dist)
    n2_dist = np.array(n2_dist)
    cos_threshold_range = create_range(cos_dist.min(), cos_dist.max(), 1000)
    n2_threshold_range = create_range(n2_dist.min(), n2_dist.max(), 1000)

    cos_roc = cal_threshold(cos_dist, cos_threshold_range)
    n2_roc = cal_threshold(n2_dist, n2_threshold_range)

    return cos_roc, n2_roc


if __name__ == '__main__':
    cos_roc, n2_roc = main()
    print('cos_roc = ', end='[')
    for trueneg, falsepos, threahold in cos_roc:
        print(trueneg, falsepos, threahold, end=';')
    print('];')

    print('n2_roc = ', end='[')
    for trueneg, falsepos, threahold in n2_roc:
        print(trueneg, falsepos, threahold, end=';')
    print('];')
