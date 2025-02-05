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

        if split == 0:
            class_list = list(range(0, 5))  # [1,2,3,4,5]
        elif split ==1:
            class_list = list(range(5, 10))
        elif split == 2:
            class_list = list(range(10, 15))
        elif split ==3:
            class_list = list(range(15, 20))

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
                score_dict[item[0][0]] = item[1] * 0.3 + distance[i] * 0.7

            d_order = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)

            ### Visualize
            # for i,item in enumerate(d_order):
            #     img_name = item[0][0]
            #     score = item[1].item()
            #
            #     img_path = '/media/meng2/disk11/ZRT/dataset/COCO2017_unlabeled/unlabeled2017/%s'%img_name
            #     label_path = buffer_path + 'Buffer/%s'%(img_name[:-4]+'_class'+str(sub_class)+'.png')
            #
            #     trival_index = name_tmp_list.index(item[0][0])
            #     save_img = visualize(img_path, label_path)
            #     os.makedirs('visualizePreAB/class%d' % sub_class, exist_ok=True)
            #     # save_image(save_img, 'visualize/check_S_Expansion/class%d/%d_%s_D_%.5f_Q_%.5f.png' % (sub_class, i, img_name[:-4],
            #     #                                                                                       distance[trival_index], name_list[trival_index][1]))
            #     save_image(save_img, 'visualizePreAB/class%d/%d_%s_%.5f.png' % (sub_class, i, img_name[:-4], score))

            flag1=0
            flag2=0
            flag3=0
            idx1=0
            idx2=0
            idx3=0
            for idx, item in enumerate(d_order):
                if item[1] < 0.55 and flag1==0:
                    idx1=idx
                    flag1=1
                elif item[1] < 0.50 and flag2==0:
                    idx2=idx
                    flag2=1
                elif item[1] < 0.45 and flag3 == 0:
                    idx3=idx
                    flag3=1
                    break

            print(class_idx, idx1, idx2, idx3, len(d_order),'||', '%f:%f, %f:%f, %f:%f'%(idx1/(len(d_order)), 1-idx1/(len(d_order)),
                                    idx2/(len(d_order)), 1-idx2/(len(d_order)), idx3/(len(d_order)), 1-idx3/(len(d_order))))
