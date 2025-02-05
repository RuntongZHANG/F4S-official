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
        elif split ==1:
            class_list = list(range(5, 10))
        elif split == 2:
            class_list = list(range(10, 15))
        elif split ==3:
            class_list = list(range(15, 20))

        save_f = open('data/splits/pascal/wild/fold%d.txt'%split,'w')
        buffer_path = '/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/' % split
        target_path='/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split%d/'%split

        os.makedirs(target_path + 'JPEGImages', exist_ok=True)
        os.makedirs(target_path + 'PseudoLabel', exist_ok=True)

        for sub_class in class_list:
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
                score_dict[item[0][0]] = item[1]*0.3 + distance[i]*0.7
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


            ### generate S-Expansion
            for item in d_order:
                img_name = item[0][0]
                score = item[1].item()
                if score<0.65:
                    continue

                img_source = '/media/meng2/disk11/ZRT/dataset/COCO2017_unlabeled/unlabeled2017/%s'%img_name
                img_target = target_path + 'JPEGImages/%s'%img_name
                shutil.copy(img_source, img_target) # image

                label_source = buffer_path + 'Buffer/%s'%(img_name[:-4]+'_class'+str(sub_class)+'.png')
                label_target = target_path + 'PseudoLabel/%s'%img_name[:-4]+'.png'
                if os.path.exists(label_target):
                    continue
                else:
                    shutil.copy(label_source, label_target)
                    save_f.write(img_name[:-4]+'__%02d'%(sub_class+1))
                    save_f.write('\n')

        save_f.close()