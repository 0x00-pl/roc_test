import numpy as np


def read_features(path_name):
    with open(path_name) as fin:
        content = fin.read()
        str_features = content.replace('\n', '').split(',')
        features = [float(i) for i in str_features if len(i)>1]
        return np.array(features)/np.linalg.norm(features)


def read_all(idx_range, label_range, file_prefix='features_400/', file_suffix='.feature.txt', idx_len=4):
    label_feature_list = []
    for idx, label in zip(idx_range, label_range):
        file_idx = ('0' * idx_len + str(idx))[-idx_len:]
        filename = file_prefix + file_idx + file_suffix
        feature = read_features(filename)
        label_feature_list.append([label, feature])

    return label_feature_list


def cal_cos_dist(feature1, feature2):
    dots = np.dot(feature1, feature2)
    # cos_dist = dots/(np.linalg.norm(feature1)*np.linalg.norm(feature2))
    cos_dist = dots
    return cos_dist


def cal_cos_dist_trueneg_falsepos(feature1, feature2, label1, label2, threshold):
    cos_dist = cal_cos_dist(feature1, feature2)
    y = cos_dist > threshold
    if label1 == label2 and not y:
        return 1, 0
    elif label1 != label2 and y:
        return 0, 1
    else:
        return 0, 0


def cal_trueneg_falsepos(label_feature_list, threshold):
    trueneg_sum, falsepos_sum = 0, 0
    for a in range(len(label_feature_list)-1):
        for b in range(a, len(label_feature_list)):
            a_label, a_feature = label_feature_list[a]
            b_label, b_feature = label_feature_list[b]
            trueneg, falsepos = cal_cos_dist_trueneg_falsepos(a_feature, b_feature, a_label, b_label, threshold)
            trueneg_sum = trueneg_sum+trueneg
            falsepos_sum = falsepos_sum+falsepos

    return trueneg_sum, falsepos_sum, threshold


def cal_trueneg_falsepos_nearest_100_400(label_feature_list, threshold):
    target_feature_list = []
    for idx in range(100):
        target2_label, target2_feature = label_feature_list[idx+100]
        target3_label, target3_feature = label_feature_list[idx+200]
        target4_label, target4_feature = label_feature_list[idx+300]
        feature_avg = (np.array(target2_feature) + np.array(target3_feature) + np.array(target4_feature))/3
        target_feature_list.append([target2_label, feature_avg])

    trueneg_sum, falsepos_sum = 0, 0
    for testing_idx in range(100):
        for target_idx in range(100):
            testing_label, testing_feature = label_feature_list[testing_idx]
            target_label, target_feature = target_feature_list[target_idx]
            trueneg, falsepos = cal_cos_dist_trueneg_falsepos(testing_feature, target_feature, testing_label, target_label, threshold)
            trueneg_sum = trueneg_sum+trueneg
            falsepos_sum = falsepos_sum+falsepos

    return trueneg_sum, falsepos_sum, threshold


def create_range(minv, maxv, countv=100):
    props = np.array(range(countv))/countv
    return minv + props * (maxv-minv)


def print_roc(roc_list, name='roc_cos'):
    print(name, '=', end='[')
    for trueneg, falsepos, threahold in roc_list:
        print(trueneg, falsepos, threahold, end=';')
    print('];')


def main():
    label_list = list(range(100))*4
    label_feature_list = read_all(range(1, 401), label_list)

    threshold_list = create_range(0, 1, 1000)

    cos_dist_roc = []
    for threshold in threshold_list:
        print('threshold', threshold)
        trueneg, falsepos, th = cal_trueneg_falsepos_nearest_100_400(label_feature_list, threshold)
        cos_dist_roc.append([trueneg, falsepos, th])

    print_roc(cos_dist_roc)


if __name__ == '__main__':
    main()


