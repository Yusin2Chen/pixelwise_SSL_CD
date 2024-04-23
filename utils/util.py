import os
import random
from torch import nn
from PIL import ImageFilter
import torch
import numpy as np
from kornia import augmentation as augs
from kornia import filters
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

class pix_to_superpix_cuda():
    def __init__(self,):
        pass
    def unique(self, x, dim=0):
        """Unique elements of x and indices of those unique elements
        https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

        e.g.

        unique(tensor([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [1, 2, 5]
        ]), dim=0)
        => (tensor([[1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5]]),
            tensor([0, 1, 3]))
        """
        unique, inverse = torch.unique(
            x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                            device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

    def get_spix_data(self, segments, outFeats):
        self.segments = segments
        self.outFeats = outFeats
        self.value, _ = self.unique(segments.view(1, -1))

        for i in self.value:
            s0 = self.segments == i
            ex_dim_s0 = s0[:, None, :, :]
            mask_nums = s0.sum(axis=1).sum(axis=1)
            mask_nums[mask_nums == 0] = 1
            mask_nums = mask_nums[:, None]
            masked = ex_dim_s0 * self.outFeats
            sum_sup_feats = masked.sum(axis=2).sum(axis=2)
            avg_sup_feats = sum_sup_feats / mask_nums
            self.outFeats[s0] = avg_sup_feats
        return self.outFeats

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def Rnormalize_S2(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i,:,:] * S2_STD[i]) + S2_MEAN[i]
    return imgs


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def default(val, def_val):
    return def_val if val is None else val

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# default SimCLR augmentation
image_size = 256
DEFAULT_AUG = nn.Sequential(
            RandomApply(augs.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            augs.RandomGrayscale(p=0.2),
            augs.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            augs.RandomResizedCrop((image_size, image_size)))
            #color.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def labels_to_dfc(tensor, no_savanna):
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class
    if no_savanna:
        for i in range(2, 9):
            out[tensor == i] = i + 1
    else:
        pass

    # transform from zero-based labels to 1-10
    out[tensor != 255] += 1

    # make sure the mask is intact and return transformed labels
    assert np.all((tensor == 255) == (out == 255))
    return out


def display_input_batch(tensor, display_indices=0, brightness_factor=3):

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor, no_savanna=False):

    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    tensor = labels_to_dfc(tensor, no_savanna)

    # colorize labels
    cmap = mycmap()
    imgs = []
    for s in range(tensor.shape[0]):
        im = (tensor[s, :, :] - 1) / 10
        im = cmap(im)[:, :, 0:3]
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def classnames():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]


def mycmap():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#c24f44',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap


def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches

def seed_torch(seed=1029):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def visualization(prediction, target, ID, image, args): # ID-> batch['id']
    id_list = ['ROIs0000_autumn_dfc_BandarAnzali_p611.tif', 'ROIs0000_autumn_dfc_CapeTown_p1277.tif',
               'ROIs0000_autumn_dfc_Mumbai_p182.tif', 'ROIs0000_autumn_dfc_Mumbai_p255.tif',
               'ROIs0000_spring_dfc_BlackForest_p873.tif', 'ROIs0000_winter_dfc_MexicoCity_p440.tif',
               'ROIs0000_winter_dfc_KippaRing_p268.tif', 'ROIs0000_autumn_dfc_CapeTown_p63.tif',
               'ROIs0000_autumn_dfc_Mumbai_p256.tif', 'ROIs0000_autumn_dfc_Mumbai_p444.tif']
    args.score = True
    if not os.path.isdir(args.preview_dir):
        os.makedirs(args.preview_dir)
    # back normlize image
    image = Rnormalize_S2(image)
    image /= 10000
    # convert to 256x256 numpy arrays
    prediction = prediction.cpu().numpy()
    prediction = np.argmax(prediction, axis=1)
    if args.score:
        target = target.cpu().numpy()

    # save predictions
    gt_id = "dfc"
    for i in range(prediction.shape[0]):

        # n += 1
        #id = ID[i].replace("_s2_", "_" + gt_id + "_")
        id = ID[i]

        if id in id_list:
            output = labels_to_dfc(prediction[i, :, :], args.no_savanna)

            output = output.astype(np.uint8)
            #output_img = Image.fromarray(output)
            #output_img.save(os.path.join(args.out_dir, id))

            # update error metrics
            if args.score:
                gt = labels_to_dfc(target[i, :, :], args.no_savanna)
                #conf_mat.add(target[i, :, :], prediction[i, :, :])

            # save preview
            if args.preview_dir is not None:

                # colorize labels
                cmap = mycmap()
                output = (output - 1) / 10
                output = cmap(output)[:, :, 0:3]
                if args.score:
                    gt = (gt - 1) / 10
                    gt = cmap(gt)[:, :, 0:3]
                display_channels = [2, 1, 0]
                brightness_factor = 3

                if args.score:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                if image.shape[1] > 3:
                    img = image.cpu().numpy()[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                else:
                    img = image.cpu().numpy()[i, -2:-1, :, :]
                    img = np.rollaxis(img, 0, 3)
                ax1.imshow(np.clip(img * brightness_factor, 0, 1))
                ax1.set_title("input")
                ax1.axis("off")
                ax2.imshow(output)
                ax2.set_title("prediction")
                ax2.axis("off")
                if args.score:
                    ax3.imshow(gt)
                    ax3.set_title("label")
                    ax3.axis("off")
                lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0), handles=mypatches(), ncol=2,
                                 title="DFC Classes")
                ttl = fig.suptitle(id, y=0.75)
                plt.savefig(os.path.join(args.preview_dir, id),
                            bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
                plt.close()

#thresholding methods
def kde_statsmodels_u(x, x_grid, bandwidth, **kwargs):
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)

    #Rosin
def rosin(heatmap, maxPercent = 5):
    heatmap_list = heatmap.flatten().tolist()
    new_data = np.array(heatmap_list)
    #new_data = heatmap.flatten()
    #new_data = f_heatmap - np.min(f_heatmap) + 0.001
    # declare kernel estimation parameters
    bandwidth = 0.06
    # estimate kernel
    x_grid = np.linspace(0, np.max(new_data), 90)  # x-coordinates for data points in the kernel
    kernel = kde_statsmodels_u(new_data, x_grid, bandwidth)  # get kernel

    # get the index of the kernal peak
    maxIndex = np.argmax(kernel)

    # Assign percent below the max kernel value for the 'zero' peak i.e. a value of 2 = 2% the maximum value
    #maxPercent = 5

    # assign x and y coords for peak-to-base line
    x1 = x_grid[maxIndex]
    y1 = kernel[maxIndex]
    # find all local minima in the kernel
    local_mins = np.where(np.r_[True, kernel[1:] < kernel[:-1]] & np.r_[kernel[:-1] < kernel[1:], True])
    local_mins = local_mins[0]  # un 'tuple' local mins
    # filter for points below a certain kernel max
    local_mins = local_mins[(np.where(kernel[local_mins] < (y1 / (100 / maxPercent))))]
    # get local minima beyond the peak
    local_mins = local_mins[(np.where(local_mins > maxIndex))]  # get local minima that meet percent max threshold
    x2_index = local_mins[0]  # find minumum beyond peak of kernel
    x2 = x_grid[x2_index]  # index to local min beyond kernel peak
    y2 = kernel[x2_index]

    # calculate line slope and get perpendicular line
    slope = (y2 - y1) / (x2 - x1)
    # find y_intercept for line
    y_int = y1 - (slope * x1)
    slopeTan = -1 / slope  # perpendicular line slope

    # allocate lists for x-y coordinates and distance values
    dist = list()
    # save x-y coords of intersect points
    yii = list()
    xii = list()

    # iterate and generate perpendicular lines
    for i in range(maxIndex + 1, x2_index):
        # find intersection point between lines
        # determine equation of the perpendicular line based on current bin coordinate
        xt1 = x_grid[i]
        yt1 = kernel[i]
        y_int_tan = yt1 - (slopeTan * xt1)
        # calculate intersection point between lines
        b1 = y_int
        b2 = y_int_tan
        m1 = slope
        m2 = slopeTan
        # y = mx + b
        # Set both lines equal to find the intersection point in the x direction, y1=y2, x1=x2
        # y1 = m1 * x + b1, y2 = m2 * x + b2
        # if y1 == y2...
        # m1 * x + b1 = m2 * x + b2
        # m1 * x - m2 * x = b2 - b1
        # x * (m1 - m2) = b2 - b1
        # x = (b2 - b1) / (m1 - m2)
        xi = (b2 - b1) / (m1 - m2)
        # Now solve for y -- use either line, because they are equal here
        # y = mx + b
        yi = m1 * xi + b1
        # assert that the new line generated is equal or very close to the correct perpendicular value of the max deviation line
        assert ((m2 - m2 * .01) < ((yi - y_int_tan) / (xi - 0)) < (
                    m2 + m2 * .01))  # an error will throw if this statement is false
        # save x-y coordinates of the point
        yii.append(yi)
        xii.append(xi)
        # get euclidean distance between kernel coordinate and intersect point
        euc = np.sqrt((xi - xt1) ** 2 + (yi - yt1) ** 2)
        # store the euclidean distance
        dist.append(euc)

    # get kernel point with the maximum distance from the Rosin line
    # remeber, we started at maxIndex+1, so the index of the optimalPoint in the kernel array will be maxIndex+1
    # + the index in the 'dist' array
    optimalPoint = np.argmax(dist) + maxIndex + 1
    # plot the optimal point over the kernel with Rosin line we plotted before
    threshold = x_grid[optimalPoint]
    #final_threhold = threshold + np.min(f_heatmap)
    #return heatmap < final_threhold
    return threshold


# 5. Rosin Thresholding
def Rosin_Threshold(signal):
    x = signal.flatten()
    bins = int(len(x) / 10)  # 1/10 size of data
    hists, bin_edges = np.histogram(x, bins=bins)

    # Histogram peak coordinate Xp, Yp
    Xp = hists.argmax()
    Yp = hists.max()

    # Histogram non-zero end coordinate Xe, Ye
    Xe = np.where(hists > 0)[0][-1]
    Ye = hists[Xe]

    # Assign start values for best threshold finding
    best_idx = -1
    max_dist = -1

    # Find best index on histogram
    for X in range(Xp, Xe):
        Y = hists[X]
        a = [Xp - Xe, Yp - Ye]
        b = [X - Xe, Y - Ye]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if dist > max_dist:
            best_idx = X
            max_dist = dist

    # Calculate threshold with bin_edge values
    Threshold = 0.5 * (bin_edges[best_idx] + bin_edges[best_idx + 1])

    return Threshold
