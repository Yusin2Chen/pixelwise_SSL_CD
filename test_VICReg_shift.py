"""
DDP training for Contrastive Learning
"""
from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from datasets.datasets_infer import DFC2020
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from models.ResUnet_cls import twinshift
from matplotlib import colors

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def Rnormalize_S2(imgs):
    for i in range(4):
        imgs[:,i,:,:] = (imgs[:,i,:,:] * S2_STD[i]) + S2_MEAN[i]
    return imgs

def parse_option():

    parser = argparse.ArgumentParser('argument for test')
    # specify folder
    parser.add_argument('--data_folder', type=str, default='./data/SpaceNet7', help='path to data')
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='./tb', help='path to tensorboard')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=64, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # output
    parser.add_argument('--out_dir', type=str, default='./result_SpaceNet7_shift2_InferS2_all_VICReg', help='path to save linear classifier')
    parser.add_argument('--score', action='store_true', default=True, help='score prediction results using ground-truth data')
    parser.add_argument('--preview_dir', type=str, default='./preview_SpaceNet7_shift2_InferS2_all_VICReg', help='path to preview dir (default: no previews)')


    opt = parser.parse_args()

    if not os.path.isdir(opt.out_dir):
        os.makedirs(opt.out_dir)

    if not os.path.isdir(opt.preview_dir):
        os.makedirs(opt.preview_dir)

    return opt

def get_train_loader(args):
    # load datasets
    train_set = DFC2020(args.data_folder,
                        subset="train",
                        no_savanna=args.no_savanna,
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        unlabeled=True,
                        transform=False,
                        crop_size=128)
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs
    n_samples = len(train_set)
    args.no_savanna = train_set.no_savanna
    args.display_channels = train_set.display_channels
    args.brightness_factor = train_set.brightness_factor

    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, n_inputs, n_classes, n_samples

def encoder_factory(model, args):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels: the number of output channels
    """
    # load pre-trained model
    print('==> loading pre-trained model')
    pretrained_model = os.path.join('./save_VIReg_shift_spixTS/resunet18_crop_32_fetdim_128', 'twins_epoch_229_37.44693374633789.pth')
    #pretrained_model = os.path.join('./save_twins2s2_shift_pixTS/resunet18_crop_32_fetdim_128', 'twins_epoch_99_10.73639965057373.pth')
    ckpt = torch.load(pretrained_model)
    pretrained_dict = ckpt['online_network_state_dict']
    model_dict = model.state_dict()
    # filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model.cuda()
    #return model

def non_zero_mean(cd, crop):
    exist = (cd != 0)
    if (exist.sum() != 0):
        dist = (crop - cd) * exist
        dist = dist.sum() / exist.sum()
        crop = crop - dist
    np_arr = np.dstack([cd, crop])
    num = np_arr.sum(axis=2)
    den = exist + 1
    return num/den


def Crop_img(img, CropSize, RepetitionRate, model, args):
    batch, bands, height, width = img.shape
    cd_img = np.zeros((height, width))
    # 裁剪图片,重复率为RepetitionRate
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            # 如果图像是单波段
            if (len(img.shape) == 3):
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            # 如果图像是多波段
            else:
                cropped = img[:, :,
                          int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            #to gpu and output
            if args.use_gpu:
                cropped = cropped.cuda()
            cropped_1, cropped_2 = torch.split(cropped, [4, 4], dim=1)
            surrogate_label = model(cropped_1, mode=1)
            pred = model(cropped_2, mode=1)
            #prediction = np.linalg.norm(pred.cpu().detach().numpy() - surrogate_label.cpu().detach().numpy(), axis=1)
            #prediction = np.squeeze(prediction)
            prediction = -1 * torch.cosine_similarity(pred, surrogate_label, dim=1) + 1
            prediction = prediction.squeeze().cpu().numpy()
            cd = cd_img[
            int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
            int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
            cd_img[
            int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
            int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize] = non_zero_mean(cd, prediction)

    # 向前裁剪最后一列
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 3):
            cropped = img[:, int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize):width]
        else:
            cropped = img[:, :,
                      int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                      (width - CropSize):width]
        # to gpu and output
        if args.use_gpu:
            cropped = cropped.cuda()
        cropped_1, cropped_2 = torch.split(cropped, [4, 4], dim=1)
        surrogate_label = model(cropped_1, mode=1)
        pred = model(cropped_2, mode=1)
        #prediction = np.linalg.norm(pred.cpu().detach().numpy() - surrogate_label.cpu().detach().numpy(), axis=1)
        #prediction = np.squeeze(prediction)
        prediction = -1 * torch.cosine_similarity(pred, surrogate_label, dim=1) + 1
        prediction = prediction.squeeze().cpu().numpy()
        cd = cd_img[int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
        (width - CropSize):width]
        cd_img[int(i * CropSize * (1 - RepetitionRate)):int(i * CropSize * (1 - RepetitionRate)) + CropSize,
        (width - CropSize):width] = non_zero_mean(cd, prediction)
    # 向前裁剪最后一行
    for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        if (len(img.shape) == 3):
            cropped = img[:, (height - CropSize):height,
                      int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        else:
            cropped = img[:, :, (height - CropSize):height,
                      int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        # to gpu and output
        if args.use_gpu:
            cropped = cropped.cuda()
        cropped_1, cropped_2 = torch.split(cropped, [4, 4], dim=1)
        surrogate_label = model(cropped_1, mode=1)
        pred = model(cropped_2, mode=1)
        # prediction = np.linalg.norm(pred.cpu().detach().numpy() - surrogate_label.cpu().detach().numpy(), axis=1)
        # prediction = np.squeeze(prediction)
        prediction = -1 * torch.cosine_similarity(pred, surrogate_label, dim=1) + 1
        prediction = prediction.squeeze().cpu().numpy()
        cd = cd_img[(height - CropSize):height,
        int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize]
        cd_img[(height - CropSize):height,
        int(j * CropSize * (1 - RepetitionRate)):int(j * CropSize * (1 - RepetitionRate)) + CropSize] = non_zero_mean(cd, prediction)
    #  裁剪右下角
    if (len(img.shape) == 3):
        cropped = img[:, (height - CropSize): height, (width - CropSize): width]
    else:
        cropped = img[:, :, (height - CropSize): height, (width - CropSize): width]
    # to gpu and output
    if args.use_gpu:
        cropped = cropped.cuda()
    cropped_1, cropped_2 = torch.split(cropped, [4, 4], dim=1)
    surrogate_label = model(cropped_1, mode=1)
    pred = model(cropped_2, mode=1)
    # prediction = np.linalg.norm(pred.cpu().detach().numpy() - surrogate_label.cpu().detach().numpy(), axis=1)
    # prediction = np.squeeze(prediction)
    prediction = -1 * torch.cosine_similarity(pred, surrogate_label, dim=1) + 1
    prediction = prediction.squeeze().cpu().numpy()
    cd = cd_img[(height - CropSize): height, (width - CropSize): width]
    cd_img[(height - CropSize): height, (width - CropSize): width] = non_zero_mean(cd, prediction)

    return cd_img


def validate(val_loader, classifier, args):
    """
    evaluation
    """
    # switch to evaluate mode
    classifier.eval()

    # main validation loop
    #conf_mat = metrics.ConfMatrix(args.n_classes, args.crop_size)

    with torch.no_grad():
        for idx, (batch) in enumerate(val_loader):

            args.score=False
            # unpack sample
            if args.score:
                image, target = batch['image'], batch['label']
            else:
                image = batch['image']
            # crop and output
            cd_img = Crop_img(image, 512, 0.0, classifier, args) #128 for small  / 256 for big
            #norm = colors.Normalize(vmin=0, vmax= cd_img.mean() + 3 * cd_img.std())

            if args.score:
                target = target.cpu().numpy()
                target = np.squeeze(target)
            pre_img, pos_img = torch.split(image, [4, 4], dim=1)
            pre_img = Rnormalize_S2(pre_img)
            pre_img /= 10000
            pos_img = Rnormalize_S2(pos_img)
            pos_img /= 10000
            pre_img = pre_img.cpu().numpy()
            pre_img = np.squeeze(pre_img)
            pos_img = pos_img.cpu().numpy()
            pos_img = np.squeeze(pos_img)


            # save predictions
            gt_id = "cim"
            id = batch["id"][0]
            id = id.replace("_as2_", "_" + gt_id + "_")
            ##output = cd_img.astype(np.uint8)
            output_img = Image.fromarray(cd_img)
            output_img.save(os.path.join(args.out_dir, id))

            # save preview
            if args.preview_dir is not None:

                display_channels = [2, 1, 0]
                brightness_factor = 3
                plt.rcParams['figure.dpi'] = 300
                if args.score:
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                else:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                pre_img = pre_img[display_channels, :, :]
                pre_img = np.rollaxis(pre_img, 0, 3)
                pos_img = pos_img[display_channels, :, :]
                pos_img = np.rollaxis(pos_img, 0, 3)
                ax1.imshow(np.clip(pre_img * brightness_factor, 0, 1))
                ax1.set_title("pre")
                ax1.axis("off")
                ax2.imshow(np.clip(pos_img * brightness_factor, 0, 1))
                ax2.set_title("post")
                ax2.axis("off")
                ax3.imshow(cd_img)
                ax3.set_title("prediction")
                ax3.axis("off")
                if args.score:
                    ax4.imshow(target)
                    ax4.set_title("label")
                    ax4.axis("off")
                plt.savefig(os.path.join(args.preview_dir, id), bbox_inches='tight')
                plt.close()

def main(args):
    # set flags for GPU processing if available
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False

    # build model
    online_network = twinshift(width=1, in_channel=4, in_dim=128, feat_dim=128)
    model = encoder_factory(online_network, args)

    # build dataset
    train_loader, n_inputs, n_classes, n_samples = get_train_loader(args)

    # inference
    validate(train_loader, model, args)


if __name__ == '__main__':
    args = parse_option()
    main(args)
