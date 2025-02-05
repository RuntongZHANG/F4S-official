r""" Hypercorrelation Squeeze testing code """
import os
import argparse
import random
import time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
sys.path.append("/data1/zhang_runtong/projects/hsnet-main")

from model.hsnet import HypercorrSqueezeNetwork
from common.logger import Logger, AverageMeter
from common.vis import Visualizer
from common.evaluation import Evaluator
from common import utils
from data.dataset import FSSDataset_SExpansion

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def test(model, dataloader, nshot, seed):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    #utils.fix_randseed(0)
    utils.fix_randseed(seed)
    average_meter = AverageMeter(dataloader.dataset)

    res = []

    for idx, batch in enumerate(dataloader):
        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = utils.to_cuda(batch)

        torch.cuda.synchronize()
        start = time.time()
        pred_mask, _, _, _ = model.module.predict_mask_nshot(batch, nshot=nshot)
        torch.cuda.synchronize()
        end = time.time()
        res.append(end-start)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

        # Visualization
        # save_path = 'visualize/test-expansion1/'
        # os.makedirs(save_path, exist_ok=True)
        # utils.visualize(batch['support_imgs'][0], batch['support_masks'][0], batch['query_img'][0], pred_mask,
        #                 save_name=os.path.join(save_path, '%s.png' % (idx)))

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou, miou_bg, miou_fg_list = average_meter.compute_iou()

    time_sum = 0
    for i in res:
        time_sum += i
    print('FPS:%f'%(1.0/(time_sum/len(res))))
    print('time:%f'%(time_sum/len(res)))

    return miou, fb_iou, miou_bg, miou_fg_list


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Hypercorrelation Squeeze Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default='../Datasets_HSN')
    parser.add_argument('--benchmark', type=str, default='pascal_expansion', choices=['pascal', 'coco', 'fss'])
    parser.add_argument('--logpath', type=str, default='exp1')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    # parser.add_argument('--load', type=str, default='checkpoints/pascal/res101_pas/res101_pas_fold0/best_model.pt')
    # parser.add_argument('--load', type=str, default='logs/ourtrain_split0.log/best_model.pt')
    parser.add_argument('--load', type=str, default='logs/pascal_ourtrain_PR_resnet101_split0.log/best_model.pt')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--nexpansion', type=int, default=4)
    parser.add_argument('--backbone', type=str, default='resnet101', choices=['vgg16', 'resnet50', 'resnet101'])
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--use_original_imgsize', action='store_true')
    args = parser.parse_args()
    Logger.initialize(args, training=False)

    # Model initialization
    model = HypercorrSqueezeNetwork(args.backbone, args.use_original_imgsize)
    model.eval()
    Logger.log_params(model)

    # flops, params = profile(model, inputs=torch.ones((1,1,3,400,400)))
    # print('flops:'+str(flops/1000**3)+'G')
    # print('params:'+str(params/1000**2)+'M')

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    if args.load == '': raise Exception('Pretrained model not specified.')
    model.load_state_dict(torch.load(args.load))

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    FSSDataset_SExpansion.initialize(img_size=400, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    dataloader_test = FSSDataset_SExpansion.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot, args.nexpansion)

    # Test HSNet
    seed_list=[]
    miou_list=[]
    fb_miou_list=[]
    for i in range(5):
        seed = int(random.randint(0,999))
        with torch.no_grad():
            test_miou, test_fb_iou, _, _ = test(model, dataloader_test, args.nshot+args.nexpansion, seed)
        seed_list.append(seed)
        miou_list.append(test_miou.item())
        fb_miou_list.append(test_fb_iou.item())
        Logger.info('Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (args.fold, test_miou.item(), test_fb_iou.item()))
        Logger.info('==================== Finished Testing ====================')
    print(seed_list, miou_list, fb_miou_list)

    with open('pascal_test_split0.txt','a') as f:
        f.write('%s with %d-pseudo' % (args.backbone, args.nexpansion))
        f.write('\n')
        f.write('seed: '+str(seed_list))
        f.write('\n')
        f.write('miou: '+str(miou_list)+' mean_miou:'+str(np.mean(miou_list))+' std:'+str(np.std(miou_list)))
        f.write('\n')
        f.write('fbiou: '+str(fb_miou_list)+' mean_miou:'+str(np.mean(fb_miou_list))+' std:'+str(np.std(fb_miou_list)))
        f.write('\n')