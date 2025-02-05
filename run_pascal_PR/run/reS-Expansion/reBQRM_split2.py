r""" Hypercorrelation Squeeze testing code """
import os
import pickle
import argparse
import numpy as np
from PIL import Image

import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.utils import save_image

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator_QS
from common import utils
from data.dataset import FSSDataset_WildImage_reSE

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def test(args, model, model_aux, nshot):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    utils.fix_randseed(0)

    class_list = [10,11,12,13,14]
    for class_ in class_list:
        # Dataset initialization
        FSSDataset_WildImage_reSE.initialize(img_size=400, datapath=args.datapath,
                                             use_original_imgsize=args.use_original_imgsize)
        dataloader = FSSDataset_WildImage_reSE.build_dataloader(args.benchmark, args.bsz,
                                                                args.nworker, args.fold, 'test',
                                                                args.nshot, args.nexpansion, class_sample=class_)

        s_f_list = []
        q_f_list = []
        name_list = []

        for idx, batch in enumerate(dataloader):

            # Hypercorrelation Squeeze Networks forward pass
            batch = utils.to_cuda(batch)
            pred_mask, logits, s_f, q_f = model.module.predict_mask_nshot(batch, nshot=nshot)
            pred_mask2, _, _, _ = model_aux.module.predict_mask_nshot(batch, nshot=nshot)

            # Quality Scores
            # 1.
            area_inter, area_union = Evaluator_QS.classify_prediction(pred_mask.clone(), pred_mask2.clone())
            #Eimc = torch.mean(area_inter/area_union).item()
            Eimc = area_inter[1].item() / (area_union[1].item() + 1e-7)
            # 2.
            batch_cycle = batch.copy()
            batch_cycle['query_img'] = batch.get('support_imgs')[0][:1]
            batch_cycle['query_mask'] = batch.get('support_masks')[0][:1]
            batch_cycle['support_imgs'] = batch.get('query_img')[None,...]
            batch_cycle['support_masks'] = pred_mask[None,...]
            pred_mask_cycle, _, _, _ = model.module.predict_mask_nshot(batch_cycle, nshot=nshot)
            area_inter_cycle, area_union_cycle = Evaluator_QS.classify_prediction(pred_mask_cycle.clone(), batch.get('support_masks')[0])
            #Ecyc = torch.mean(area_inter_cycle/area_union_cycle)
            Ecyc = area_inter_cycle[1].item() / (area_union_cycle[1].item() + 1e-7)
            # 3.
            logits = torch.softmax(logits, dim=1)  # torch.Size([1, 2, 473, 473])
            h = -torch.mul(logits, torch.log2(logits + 1e-7)) - torch.mul(1 - logits, torch.log2(1 - logits + 1e-7))
            h_mean = torch.mean(h)
            Esc = -h_mean + 1

            QS = Esc*(Eimc + Ecyc)/2  # the quality score

            # for SSM
            s_f = F.adaptive_avg_pool2d(s_f, (1,1))[:,:,0,0]
            s_f_list.append(s_f.detach().cpu().numpy())

            if QS > 0.0:

                # Visualization
                save_path = 'visualize/test2/'
                os.makedirs(save_path, exist_ok=True)
                utils.visualize(batch['support_imgs'][0], batch['support_masks'][0], batch['query_img'][0], pred_mask,
                                save_name=os.path.join(save_path, '%s.png' % (idx)))

                # Save Pseudo Labels
                resize_pred_mask = F.interpolate(pred_mask[:, None, ...].float(), size=q_f.size()[2:], mode='bilinear',align_corners=True)
                q_f = q_f * resize_pred_mask
                q_f = F.adaptive_avg_pool2d(q_f, (1, 1))[:, :, 0, 0]
                q_f_list.append(q_f.detach().cpu().numpy())
                name_list.append((batch.get('query_name'), QS, (Esc, Eimc, Ecyc)))

                pseudo_label = torch.zeros_like(pred_mask, dtype=torch.uint8)
                pseudo_label[pred_mask == 0] = 0
                pseudo_label[pred_mask != 0] = class_ +1
                pseudo_label = pseudo_label.cpu().numpy()[0]
                Image.fromarray(np.uint8(pseudo_label)).convert('L').save(
                    '/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split2/Buffer/%s_class%d.png'
                    % (batch.get('query_name')[0][0][:-4], class_))

            # log
            if idx % 100 == 0:
                print('Class %d: [%d/%d]' % (class_, idx, len(dataloader)))

        with open('/media/meng2/disk11/ZRT/PycharmProjects/Datasets_HSN/VOC2012/S-Expansion/split2/pascal_split2_class%d.pkl'%(class_),'wb') as f:
            pickle.dump([s_f_list, q_f_list, name_list], f)

if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal_wild_reSE', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='exp1')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--load', type=str, default='checkpoints/pascal/res101_pas/res101_pas_fold2/best_model.pt')
    parser.add_argument('--load_aux', type=str, default='checkpoints/pascal/res50_pas/res50_pas_fold2/best_model.pt')
    parser.add_argument('--fold', type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--nexpansion', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--backbone_aux', type=str, default='resnet50', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, args.use_original_imgsize)
    model_aux = HypercorrSqueezeNetwork(args.backbone_aux, args.use_original_imgsize)
    model.eval()
    model_aux.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model_aux = nn.DataParallel(model_aux)
    model.to(device)
    model_aux.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))
    model_aux.load_state_dict(torch.load(args.load_aux))

    # Helper classes (for testing) initialization
    Evaluator_QS.initialize()
    Visualizer.initialize(args.visualize)

    # Test HSNet
    with torch.no_grad():
        test(args, model, model_aux, args.nshot+args.nexpansion)
