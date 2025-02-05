r""" Helper functions """
import random

import torch
import numpy as np

from torchvision.utils import save_image


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def colorize(img, mask, type='pred'):
    img = img.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    if type=='pred':
        img[0,...] = np.where(mask == 1, 255, img[0,...])
        img[1, ...] = np.where(mask == 1, img[1, ...] * 0.7, img[1, ...])
        img[2, ...] = np.where(mask == 1, img[2, ...] * 0.7, img[2, ...])
    elif type=='support':
        img[0,...] = np.where(mask == 1,  img[0, ...] * 0.7, img[0, ...])
        img[1, ...] = np.where(mask == 1, img[1, ...] * 0.7, img[1, ...])
        img[2, ...] = np.where(mask == 1, 255, img[2, ...])
    elif type=='query':
        img[0,...] = np.where(mask == 1,  img[0, ...] * 0.7, img[0, ...])
        img[1, ...] = np.where(mask == 1, 255, img[1, ...])
        img[2, ...] = np.where(mask == 1, img[2, ...] * 0.7, img[2, ...])
    return torch.tensor(img).cuda()


def visualize(s_input, s_mask, q, q_mask, save_name): # s:tensor(shot,c,h,w), s_mask:tensor(shot, h,w), q:tensor(c,h,w), q_mask:tensor(h,w)
    save_img_list = []

    nshot = len(s_input)
    for id in range(nshot):
        support_item_save = torch.zeros_like(s_input[id]).cuda()
        support_item_save[0] = s_input[id,0] * 0.229 + 0.485
        support_item_save[1] = s_input[id,1] * 0.224 + 0.456
        support_item_save[2] = s_input[id,2] * 0.225 + 0.406
        colorize_img = colorize(support_item_save, s_mask[id])
        save_img_list.append(colorize_img)

    query_item_save = torch.zeros_like(q).cuda()
    query_item_save[0] = q[0] * 0.229 + 0.485
    query_item_save[1] = q[1] * 0.224 + 0.456
    query_item_save[2] = q[2] * 0.225 + 0.406
    colorize_query_img = colorize(query_item_save, q_mask)
    save_img_list.append(colorize_query_img)

    save_imgs = torch.cat(save_img_list, dim=2)

    save_image(save_imgs, save_name)


def visualize_custom(s_input, s_mask, q, pred, pred_1shot, q_mask, save_name): # s&q:tensor(shot,c,h,w), mask:tensor(h,w)
    save_img_list = []

    nshot = len(s_input)
    for id in range(nshot):
        support_item_save = torch.zeros_like(s_input[id]).cuda()
        support_item_save[0] = s_input[id,0] * 0.229 + 0.485
        support_item_save[1] = s_input[id,1] * 0.224 + 0.456
        support_item_save[2] = s_input[id,2] * 0.225 + 0.406
        colorize_img = colorize(support_item_save, s_mask[id], 'support')
        save_img_list.append(colorize_img)

    query_item_save = torch.zeros_like(q).cuda()
    query_item_save[0] = q[0] * 0.229 + 0.485
    query_item_save[1] = q[1] * 0.224 + 0.456
    query_item_save[2] = q[2] * 0.225 + 0.406
    colorize_query_img1 = colorize(query_item_save, q_mask, 'query')
    colorize_query_img2 = colorize(query_item_save, pred_1shot, 'pred')
    colorize_query_img3 = colorize(query_item_save, pred, 'pred')
    save_img_list.append(colorize_query_img1)
    save_img_list.append(colorize_query_img2)
    save_img_list.append(colorize_query_img3)

    save_imgs = torch.cat(save_img_list, dim=2)

    save_image(save_imgs, save_name)