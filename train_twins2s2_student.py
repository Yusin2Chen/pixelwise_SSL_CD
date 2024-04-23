import torch
import os
import argparse
import random
from torch.utils.data import DataLoader
from datasets.datasets_oscds2s2 import OSCD_S2
from models.ResUnetStd import ResUnet182
from models.ResUnetStd import twinshift
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from datasets.augmentation.augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomAffine, RandomPerspective
from datasets.augmentation.aug_params import RandomHorizontalFlip_params, RandomVerticalFlip_params, \
    RandomRotation_params, RandomAffine_params, RandomPerspective_params

def get_scheduler(optimizer, args):
    if args.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.epochs if args.T0 is None else args.T0,
            T_mult=args.Tmult,
            eta_min=args.eta_min,
        )
    elif args.lr_step == "step":
        m = [args.epochs - a for a in args.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=args.drop_gamma)
    else:
        return None

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    # 1600
    parser.add_argument('--batch_size', type=int, default=40, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=128, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # resume path
    parser.add_argument('--resume', action='store_true', default=True, help='path to latest checkpoint (default: none)')

    # learning rate
    parser.add_argument("--T0", type=int, help="period (for --lr_step cos)")
    parser.add_argument("--Tmult", type=int, default=1, help="period factor (for --lr_step cos)")
    parser.add_argument("--lr_step", type=str, choices=["cos", "step", "none"], default="step",
                        help="learning rate schedule type")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--eta_min", type=float, default=0, help="min learning rate (for --lr_step cos)")
    parser.add_argument("--adam_l2", type=float, default=1e-6, help="weight decay (L2 penalty)")
    parser.add_argument("--drop", type=int, nargs="*", default=[50, 25],
                        help="milestones for learning rate decay (0 = last epoch)")
    parser.add_argument("--drop_gamma", type=float, default=0.2, help="multiplicative factor of learning rate decay")
    parser.add_argument("--no_lr_warmup", dest="lr_warmup", action="store_false",
                        help="do not use learning rate warmup")

    # model definition
    parser.add_argument('--model', type=str, default='resunet18', choices=['CMC_mlp3614','alexnet', 'resnet'])
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # add new views
    #'/workplace/SSLS12'
    parser.add_argument('--data_folder', type=str, default='/workplace/OSCD_TS', help='path to training dataset')
    parser.add_argument('--dataset_val', type=str, default="dfc_cmc", choices=['sen12ms_holdout', '\
    dfc2020_val', 'dfc2020_test'], help='dataset to use for validation (default: sen12ms_holdout)')
    parser.add_argument('--model_path', type=str, default='./save_student_VICReg', help='path to save model')
    parser.add_argument('--save', type=str, default='./save_student_VICReg', help='path to save linear classifier')


    opt = parser.parse_args()

    # set up saving name
    opt.save_name = '{}_crop_{}_fetdim_{}'.format(opt.model, opt.crop_size, opt.feat_dim)
    opt.save_path = os.path.join(opt.save, opt.save_name)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    if (opt.data_folder is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if not os.path.isdir(opt.dataset_val):
        os.makedirs(opt.dataset_val)

    if not os.path.isdir(opt.data_folder):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))

    return opt

def get_train_loader(args):
    # load datasets
    train_set = OSCD_S2(args.data_folder,
                        subset="train",
                        no_savanna=args.no_savanna,
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        unlabeled=True,
                        transform=True,
                        train_index=None,
                        crop_size=args.crop_size)
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs
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

    return train_loader, n_inputs, n_classes


class BYOLTrainer:
    def __init__(self, args, online_network, target_network, optimizer, scheduler, device):
        self.augment_type = ['Horizontalflip', 'VerticalFlip']
        self.rot_agl = 15
        self.dis_scl = 0.2
        self.scl_sz = [0.8, 1.2]
        self.shear = [-0.2, 0.2]
        self.aug_RHF = RandomHorizontalFlip(p=1)
        self.aug_RVF = RandomVerticalFlip(p=1)
        self.aug_ROT = RandomRotation(p=1, theta=self.rot_agl, interpolation='nearest')
        self.aug_PST = RandomPerspective(p=1, distortion_scale=0.3)
        self.aug_AFF = RandomAffine(p=1, theta=None, h_trans=random.uniform(0, 0.2), v_trans=random.uniform(0, 0.2),
                                    scale=None, shear=None, interpolation='nearest')
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.feat_dim = args.feat_dim
        self.lr_warmup = args.lr_warmup_val
        self.lr = args.lr
        self.lr_step = args.lr_step

    def aug_list(self, img, model, params):
        for i in range(len(model)):
            img = model[i](img, params[i])
        return img

    @staticmethod
    def regression_loss(predict_teacher1, predict_teacher2, predict_student1, std):
        dis12 = -1 * torch.cosine_similarity(predict_teacher2.detach(), predict_student1, dim=1) + 1
        dis11 = -1 * torch.cosine_similarity(predict_teacher1.detach(), predict_student1, dim=1) + 1
        #print(dists.shape, std.shape)
        loss1 = torch.mul(torch.exp(-std), dis12)
        loss2 = std
        loss3 = dis11
        loss = loss1 + loss2 + loss3
        return loss.mean()


    def train(self, train_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            iters = len(train_loader)
            for idx, batch in enumerate(train_loader):
                if self.lr_warmup < 20:
                    lr_scale = (self.lr_warmup + 1) / 20
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.lr * lr_scale
                    self.lr_warmup += 1

                image = batch['image']
                loss = self.update(image)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                niter += 1
                train_loss += loss.item()

                if self.lr_step == "cos" and self.lr_warmup >= 20:
                    self.scheduler.step(epoch_counter + idx / iters)
            if self.lr_step == "step":
                self.scheduler.step()

            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
            # save checkpoints
            if (epoch_counter + 1) % 50 == 0:
                self.save_model(os.path.join(self.savepath, 'student_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))
            torch.cuda.empty_cache()

    def update(self, image):

        sample_num = 1
        aug_type = random.sample(self.augment_type, sample_num)
        model = []
        param = []
        if 'Horizontalflip' in aug_type:
            model.append(self.aug_RHF)
            param.append(RandomHorizontalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        if 'VerticalFlip' in aug_type:
            model.append(self.aug_RVF)
            param.append(RandomVerticalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        model.append(self.aug_AFF)
        param.append(RandomAffine_params(1.0, None, random.uniform(0.0, 0.2), random.uniform(0.0, 0.2),
                                         None, None, image.shape[0], image.shape[-2:], self.device, image.dtype))
        # split input
        batch_view_1, batch_view_2 = torch.split(image, [4, 4], dim=1)
        batch_view_1 = batch_view_1.to(self.device)
        batch_view_2 = batch_view_2.to(self.device)
        # tranforme one input view
        augmt_view_1 = self.aug_list(batch_view_1, model, param)

        # center crop make sure no zero in input
        batch_view_1 = batch_view_1[:, :, 32:96, 32:96]
        batch_view_2 = batch_view_2[:, :, 32:96, 32:96]
        augmt_view_1 = augmt_view_1[:, :, 32:96, 32:96]


        # compute key features
        with torch.no_grad():
            predict_teacher1, predict_teacher2 = self.online_network(batch_view_1, batch_view_2, mode=1)
            predict_teacher1 = self.aug_list(predict_teacher1, model, param)
            predict_teacher2 = self.aug_list(predict_teacher2, model, param)

        predict_student1, std = self.target_network(augmt_view_1, mode=0)

        # mask no-overlap
        with torch.no_grad():

            one = torch.ones_like(predict_teacher1)
            one_mask = self.aug_list(one, model, param)[:, 0, :, :]
            one_mask = one_mask.contiguous().view(-1)
            one_mask = one_mask.eq(1)
            # perform mask
            predict_teacher1 = predict_teacher1.permute(0, 2, 3, 1).contiguous()
            predict_teacher1 = predict_teacher1.view(-1, self.feat_dim)
            predict_teacher2 = predict_teacher2.permute(0, 2, 3, 1).contiguous()
            predict_teacher2 = predict_teacher2.view(-1, self.feat_dim)
            predict_teacher1 = predict_teacher1[one_mask, :]
            predict_teacher2 = predict_teacher2[one_mask, :]

        predict_student1 = predict_student1.permute(0, 2, 3, 1).contiguous()
        predict_student1 = predict_student1.view(-1, self.feat_dim)
        std = std.permute(0, 2, 3, 1).contiguous()
        std = std.view(-1, 1)
        predict_student1 = predict_student1[one_mask, :]
        std = std[one_mask, :]

        # loss calculation
        loss = self.regression_loss(predict_teacher1, predict_teacher2, predict_student1, std.squeeze())
        return loss.mean()

    def save_model(self, PATH):
        print('==> Saving...')
        state = {
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, PATH)
        # help release GPU memory
        del state


def main():

    # parse the args
    args = parse_option()

    # set flags for GPU processing if available
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    # set the data loader
    train_loader, n_inputs, n_classes = get_train_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = n_classes

    # set the model
    online_network = twinshift(width=1, in_channel=4, in_dim=128, feat_dim=128).to(device)
    # load pre-trained model if defined
    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'save_VIReg_shift_spixTS/resunet18_crop_32_fetdim_128')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'twins_epoch_319_36.005611419677734.pth')),
                                     map_location=device)
            pretrained_dict = load_params['online_network_state_dict']
            model_dict = online_network.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            online_network.load_state_dict(model_dict)
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
    #--> target model
    target_network = ResUnet182(width=1, in_channel=4).to(device)
    '''
    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'save_twins2s2_shift_cls')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'twins_epoch_999_5.356789708137512.pth')),
                                     map_location=device)
            pretrained_dict = load_params['online_network_state_dict']
            model_dict = target_network.state_dict()
            # filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            target_network.load_state_dict(model_dict)
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")
    '''
    # target encoder
    optimizer = torch.optim.Adam(target_network.parameters(), lr=3e-4)
    scheduler = get_scheduler(optimizer, args)
    args.lr_warmup_val = 0 if args.lr_warmup else 20
    trainer = BYOLTrainer(args,
                          online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          device=device)

    trainer.train(train_loader)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

