import os
import cv2
import time
from numpy import *
import numpy
import numpy as np
import pickle
import shutil
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import manifold
from torchvision.utils import save_image
from common.visualize import visualize

def get_cos_similarity(x, y):
    mean = x.mean(axis=0, keepdims=True)
    cos = cosine_similarity(mean, y)
    return abs(cos[0])

if __name__ == '__main__':
    for split in [0,1,2,3]:
        pca_dim = 512

        class_list = [0]
        weights = [(0.14, 0.86),(0.16, 0.84), (0.15, 0.85), (0.15, 0.85), (0.4, 0.6)]

        buffer_path = '/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/' % split
        target_path='/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/'%split


        start = time.time()
        for class_idx, sub_class in enumerate(class_list):
            with open('/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/pascal_split%d_class%d.pkl'%(split, split, sub_class), 'rb') as f:
                s_f, q_f, name_list = pickle.load(f)
            s_f = numpy.array(s_f)
            s_f = s_f[:, 0, :]
            if len(q_f)==0:
                continue
            q_f = numpy.array(q_f)
            q_f = q_f[:, 0, :]

            loc = len(s_f)

            distance = get_cos_similarity(s_f, q_f)

            name_tmp_list = []

            assert len(distance) == len(name_list)
            score_dict = {}
            for i, item in enumerate(name_list):
                img_name = item[0][0][0]
                QS = item[1]
                # Esc, Eimc, Ecyc = item[2]
                name_tmp_list.append(item[0][0][0])

                ### PR precomputeAB
                score_dict[item[0][0]] = item[1] * weights[class_idx][0] + distance[i] * weights[class_idx][1]
                # score_dict[item[0][0]] = item[1] * 0.01 + distance[i] * 0.99


            d_order = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)

            d_order = d_order[:10]

        end = time.time()
        time_sum = end - start
        print('time_sum:%f' % time_sum)
        print('number:%f' % len(name_list))
        print('FPS:%f' % (1.0 / (time_sum / len(name_list))))
        print('time:%f' % (time_sum / len(name_list)))