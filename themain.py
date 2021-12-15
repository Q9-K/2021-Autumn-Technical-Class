import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import roc_curve
# from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib

from PIL import Image,ImageQt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
from torchvision import transforms as T
import datasets.mvtec as mvtec
import time


# device setup

class detect(object):
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    def __init__(self):
        super().__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.openpath = None
        self.savepath = None
        self.qpixmap = None

    def parse_args(self):
        parser = argparse.ArgumentParser('PaDiM')
        parser.add_argument('--data_path', type=str, default='F:/padimprojiect/dataset/mvtec_anomaly_detection')
        parser.add_argument('--save_path', type=str, default='F:/padimprojiect/mvtec_result')
        parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
        return parser.parse_args()


    def detect_main(self):

        args = self.parse_args()

        # load model
        if args.arch == 'resnet18':
            model = resnet18(pretrained=True, progress=True)
            t_d = 448
            d = 100
        elif args.arch == 'wide_resnet50_2':
            model = wide_resnet50_2(pretrained=True, progress=True)
            t_d = 1792
            d = 550
        model.to(self.device)
        model.eval()
        random.seed(1024)
        torch.manual_seed(1024)
        if self.use_cuda:
            torch.cuda.manual_seed_all(1024)

        idx = torch.tensor(sample(range(0, t_d), d))

        # set model's intermediate outputs
        outputs = []

        def hook(module, input, output):
            outputs.append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig_img_rocauc = ax[0]
        fig_pixel_rocauc = ax[1]

        total_roc_auc = []
        total_pixel_roc_auc = []

        class_name = 'bottle'

        if True:

            # train_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=True)
            # train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
            # test_dataset = mvtec.MVTecDataset(args.data_path, class_name=class_name, is_train=False)
            # test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

            train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
            test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

            # extract train set features
            train_feature_filepath = os.path.join(args.save_path, 'temp_%s' % args.arch, 'train_%s.pkl' % class_name)
            
            # print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                    train_outputs = pickle.load(f)

            gt_list = []
            gt_mask_list = []
            test_imgs = []

            # extract test set features
            if True:
                resize=256
                cropsize=224
                transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                        T.CenterCrop(cropsize),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

                x = Image.open(self.openpath).convert('RGB')
                x = transform_x(x)
                mask = x
                test_img = x.cpu().detach().numpy()
                gt_mask_list = mask.cpu().detach().numpy()
                x = torch.unsqueeze(x, 0)
                # gt_mask_list.extend(mask.cpu().detach().numpy())
                # model prediction
                with torch.no_grad():
                    _ = model(x.to(self.device))
                # get intermediate layer outputs
                for k, v in zip(test_outputs.keys(), outputs):
                    test_outputs[k].append(v.cpu().detach())
                # initialize hook outputs
                outputs = []
            for k, v in test_outputs.items():
                test_outputs[k] = torch.cat(v, 0)
            
            # Embedding concat
            embedding_vectors = test_outputs['layer1']
            for layer_name in ['layer2', 'layer3']:
                embedding_vectors = self.embedding_concat(embedding_vectors, test_outputs[layer_name])

            # randomly select d dimension
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            
            # calculate distance matrix
            B, C, H, W = embedding_vectors.size()
            embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
            dist_list = []
            for i in range(H * W):
                mean = train_outputs[0][:, i]
                conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
                dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
                dist_list.append(dist)

            dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

            # upsample
            dist_list = torch.tensor(dist_list)
            score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                    align_corners=False).squeeze().numpy()
            
            # apply gaussian smoothing on the score map
            for i in range(score_map.shape[0]):
                score_map[i] = gaussian_filter(score_map[i], sigma=4)
            
            # Normalization
            max_score = score_map.max()
            min_score = score_map.min()
            # print(score_map)
            scores = (score_map - min_score) / (max_score - min_score)
            # print(scores)
            # scores是 numpy.ndarray类型
            # num = len(scores)
            # print(num)
            # print(type(scores))

        # get optimal threshold
            threshold = 0.5
            # gt_mask = np.asarray(gt_mask_list)
            # precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
            # a = 2 * precision * recall
            # b = precision + recall
            # f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            # threshold = thresholds[np.argmax(f1)]
            
            # segment = scores > threshold
            # segment[segment > 0] = 255
            # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
            # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
            # os.makedirs(save_dir, exist_ok=True)
            

            # fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
            # per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

            # fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
            # save_dir = args.save_path + '/' + f'pictures_{args.arch}'
            # os.makedirs(save_dir, exist_ok=True)
            # x = torch.squeeze(x,0)
            # x = T.ToPILImage()(x).convert('RGB')
            # x.save('1.png')
            self.plot_fig(test_img, scores, gt_mask_list, threshold)
            # x.save('1.png')


    def plot_fig(self,test_img, scores, gts, threshold):
        num = len(scores)
        vmax = scores.max() * 255.
        vmin = scores.min() * 255.
        img = self.denormalization(test_img)
        # gt = gts.transpose(1, 2, 0).squeeze()
        # heat_map = scores * 255
        mask = scores
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 2, figsize=(10, 5))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        # ax_img[1].imshow(gt, cmap='gray')
        # ax_img[1].title.set_text('GroundTruth')
        # ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        # ax_img[2].imshow(img, cmap='gray', interpolation='none')
        # ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        # ax_img[2].title.set_text('Predicted heat map')
        # ax_img[3].imshow(mask, cmap='gray')
        # ax_img[3].title.set_text('Predicted mask')
        ax_img[1].imshow(vis_img)
        ax_img[1].title.set_text('Segmentation result')
        # left = 0.92
        # bottom = 0.15
        # width = 0.015
        # height = 1 - 2 * bottom
        # rect = [left, bottom, width, height]
        # cbar_ax = fig_img.add_axes(rect)
        # cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        # cb.ax.tick_params(labelsize=8)
        # font = {
        #     'family': 'serif',
        #     'color': 'black',
        #     'weight': 'normal',
        #     'size': 8,
        # }
        # cb.set_label('Anomaly Score', fontdict=font)
        fig_img.savefig('temp.png')
        # pic = Image.open('temp.png').convert('RGB')
        # self.qpixmap = ImageQt.toqpixmap(pic)
        plt.close()
        # return fig_img


    def denormalization(self,x):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
        
        return x


    def embedding_concat(self,x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z


# if __name__ == '__main__':
#     main()