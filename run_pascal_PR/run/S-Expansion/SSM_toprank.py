import os
import cv2
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
            weights = [(0.14, 0.86),(0.19, 0.81), (0.20, 0.80), (0.22, 0.78), (0.11, 0.89)]
        elif split ==1:
            class_list = list(range(5, 10))
            weights = [(0.27, 0.73), (0.25, 0.75), (0.16, 0.84), (0.10, 0.90), (0.26, 0.74)]
        elif split == 2:
            class_list = list(range(10, 15))
            weights = [(0.21, 0.79), (0.27, 0.73), (0.26, 0.74), (0.18, 0.82), (0.34, 0.66)]
        elif split ==3:
            class_list = list(range(15, 20))
            weights = [(0.12, 0.88), (0.29, 0.71), (0.17, 0.83), (0.30, 0.70), (0.24, 0.76)]

        save_f = open('data/splits/pascal/wild/top-rank-precomputeAB-10/fold%d.txt'%split,'w')
        buffer_path = '/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/' % split
        target_path='/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/'%split

        os.makedirs(target_path + 'JPEGImages', exist_ok=True)
        os.makedirs(target_path + 'PseudoLabel_precomputeAB', exist_ok=True)

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
            # input = np.concatenate((s_f,q_f),axis=0)
            # pca = PCA(n_components=pca_dim)
            # pca = pca.fit(input)
            # output = pca.transform(input)
            #
            # s_f = output[:loc]
            # q_f = output[loc:]

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

                ### CVPR version
                # score_dict[item[0][0]] = item[1]*0.3 + distance[i]*0.7 # full
                # score_dict[item[0][0]] = item[1]
                # score_dict[item[0][0]] = distance[i] # 0001
                # score_dict[item[0][0]] = item[2][0]*0.3 + distance[i] * 0.7  # 1001
                # score_dict[item[0][0]] = item[2][1] * 0.3 + distance[i] * 0.7  # 0101
                # score_dict[item[0][0]] = item[2][2] * 0.3 + distance[i] * 0.7  # 0011

                ### ICIP version
                # score_dict[item[0][0]] = distance[i]  # 001
                # score_dict[item[0][0]] = item[2][0] * 0.3 + distance[i] * 0.7  # 101
                # score_dict[item[0][0]] = item[2][1] * 0.3 + distance[i] * 0.7  # 011
                # score_dict[item[0][0]] = item[2][0] * item[2][1] * 0.3 + distance[i] * 0.7  # 111

                ### Esc_Eimc version (alternative ICIP)
                #score_dict[item[0][0]] = item[2][1] * 0.5 + item[2][0] * 0.5


            d_order = sorted(score_dict.items(), key=lambda x:x[1], reverse=True)

            # ### Visualize
            # for i,item in enumerate(d_order):
            #     img_name = item[0][0]
            #     score = item[1].item()
            #     if score < 0.65:
            #         continue
            #
            #     img_path = '/media/meng2/disk11/ZRT/dataset/COCO2017_unlabeled/unlabeled2017/%s'%img_name
            #     label_path = buffer_path + 'Buffer/%s'%(img_name[:-4]+'_class'+str(sub_class)+'.png')
            #
            #     trival_index = name_tmp_list.index(item[0][0])
            #     save_img = visualize(img_path, label_path)
            #     os.makedirs('visualize/check_S_Expansion/class%d' % sub_class, exist_ok=True)
            #     # save_image(save_img, 'visualize/check_S_Expansion/class%d/%d_%s_D_%.5f_Q_%.5f.png' % (sub_class, i, img_name[:-4],
            #     #                                                                                       distance[trival_index], name_list[trival_index][1]))
            #     save_image(save_img, 'visualize/check_S_Expansion/class%d/%d_%s_%.5f.png' % (sub_class, i, img_name[:-4], score_dict[item[0]]))

            d_order = d_order[:10]
            #d_order = d_order

            # ### generate S-Expansion   only generate txt
            # for item in d_order:
            #     img_name = item[0][0]
            #     score = item[1].item()
            #     # if score<0.65:
            #     #     continue
            #
            #     label_source = buffer_path + 'Buffer/%s'%(img_name[:-4]+'_class'+str(sub_class)+'.png')
            #     label_target = target_path + 'PseudoLabel/%s'%img_name[:-4]+'.png'
            #     save_f.write(img_name[:-4]+'__%02d'%(sub_class+1))
            #     save_f.write('\n')

            ### generate S-Expansion
            for item in d_order:
                img_name = item[0][0]
                score = item[1]#.item()

                label_source = buffer_path + 'Buffer/%s' % (img_name[:-4] + '_class' + str(sub_class) + '.png')
                label_target = target_path + 'PseudoLabel_precomputeAB/%s' % img_name[:-4] + '.png'
                shutil.copy(label_source, label_target)

                save_f.write(img_name[:-4] + '__%02d' % (sub_class + 1))
                save_f.write('\n')

        save_f.close()