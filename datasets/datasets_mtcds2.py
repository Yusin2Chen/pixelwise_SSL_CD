import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import torch.utils.data as data

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def normalize_S2(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - S2_MEAN[i]) / S2_STD[i]
    return imgs

S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def normalize_S1(imgs):
    for i in range(2):
        imgs[i,:,:] = (imgs[i, :,:] - S1_MEAN[i]) / S1_STD[i]
    return imgs

L8_MEAN = np.array([0.13946152, 0.12857966, 0.12797806, 0.23040992])
L8_STD  = np.array([0.01898952, 0.02437881, 0.03323532, 0.04915179])

def normalize_L8(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - L8_MEAN[i]) / L8_STD[i]
    return imgs

# data augmenttaion
class RandomCrop(object):
    """给定图片，随机裁剪其任意一个和给定大小一样大的区域.

    Args:
        output_size (tuple or int): 期望裁剪的图片大小。如果是 int，将得到一个正方形大小的图片.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, unlabeld=True, superpixel=False):

        if unlabeld:
            image, id = sample['image'], sample['id']
            lc = None
        else:
            image, lc, id = sample['image'], sample['label'], sample['id']

        _, h, w = image.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h, left: left + new_w]

        #if superpixel:
        #    segments = sample["segments"]
        #    segments = segments[top: top + new_h, left: left + new_w]
        #else:
        #    segments = None

        # load label
        if unlabeld:
            #return {'image': image, 'segments': segments, 'id': id}
            return {'image': image, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            #return {'image': image, 'segments': segments, 'label': lc, 'id': id}
            return {'image': image, 'label': lc, 'id': id}


# mapping from igbp to dfc2020 classes
DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    6,  # 12 --> 6
    7,  # 13 --> 7
    6,  # 14 --> 6
    8,
    9,
    10
    ]

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]

L8_BANDS_HR = [2, 3, 4, 5]
L8_BANDS_MR = [5, 6, 7, 9, 12, 13]
L8_BANDS_LR = [1, 10, 11]

# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 = s2[:,0:512,0:512]
    s2 = normalize_S2(s2)
    return s2

def load_l8(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + L8_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + L8_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + L8_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        l8 = data.read(bands_selected)
    l8 = l8.astype(np.float32)
    l8 = np.clip(l8, 0, 1)
    l8 = normalize_L8(l8)
    return l8

# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read([1, 2])
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 = normalize_S1(s1)
    return s1

# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        lc = lc.astype(np.int64)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc[lc == 3] = 0
        lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc


# util function for reading data from single sample
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr,
                no_savanna=False, igbp=True, unlabeled=False, superpixel=False):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
    else:
        img = None

    # load s1 data
    if use_s1:
        if use_s2:
            img = np.concatenate((img, load_s2(sample["s1"], use_s2hr, use_s2mr, use_s2lr)), axis=0)
        else:
            img = load_s2(sample["s1"], use_s2hr, use_s2mr, use_s2lr)

    # segmentate the image to superpixels
    if superpixel:
        segments = np.load(sample["se"])
    else:
        segments = None

    # load label
    if unlabeled:
        return {'image': img, 'segments': segments, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
        return {'image': img, 'segments': segments, 'label': lc, 'id': sample["id"]}


# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [2, 1, 0]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)


class OSCD_S2(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 subset="val",
                 no_savanna=False,
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,
                 transform=False,
                 train_index = None,
                 crop_size = 32):
        """Initialize the dataset"""

        # inizialize
        super(OSCD_S2, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        assert subset in ["val", "train", "test"]
        self.no_savanna = no_savanna
        self.train_index = train_index
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)
        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)

        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        if subset == "train":
            train_list = []
            for seasonfolder in [
                '1286_2921_13', '2528_4620_13', '4426_3835_13', '5754_3601_13', '6752_3104_13',
                '1311_3077_13', '2569_4513_13', '4523_4210_13', '5830_3834_13', '6752_3115_13',
                '1330_3107_13', '2624_4314_13', '4553_3325_13', '5863_3800_13', '6761_3129_13',
                '1417_3281_13', '2697_3715_13', '4622_3159_13', '5912_3937_13', '6810_3478_13',
                '1487_3335_13', '2789_4694_13', '4666_2369_13', '5926_3715_13', '6813_3313_13',
                '1700_3100_13', '2832_4366_13', '4780_3377_13', '5989_3554_13', '6824_4117_13',
                '1973_3709_13', '2850_4139_13', '4791_3920_13', '6204_3495_13', '7026_3201_13',
                '2029_3764_13', '3002_4273_13', '4806_3588_13', '6353_3661_13', '7312_3008_13',
                '2065_3647_13', '3830_3914_13', '4838_3506_13', '6381_3681_13', '7367_5050_13',
                '2196_3885_13', '3998_3016_13', '4856_4087_13', '6466_3380_13', '7513_4968_13',
                '2235_3403_13', '4127_2991_13', '4881_3344_13', '6475_3361_13', '7517_4908_13',
                '2415_3082_13', '4223_3246_13', '5111_4560_13', '6678_3579_13', '8077_5007_13',
                '2459_4406_13', '4254_2915_13', '5125_4049_13', '6688_3456_13',
                '2470_5030_13', '4421_3800_13', '5195_3388_13', '6730_3430_13']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder)) if "Images" in x]
            sample_dirs = train_list
            #path = os.path.join(path, "ROIs0000_validation", "s2_validation")
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/201*.tif"), recursive=True)
            s2_locations = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]))
            # 只取一对
            if len(s2_locations) > 1:
                #s2_locations = s2_locations[0:2]
                comb_list = list(combinations(s2_locations, 2))
                # 下面这句用来筛出跨时过长的 <10
                comb_list = [comb_list[i] for i in range(len(comb_list))
                             if ((int(os.path.basename(comb_list[i][0])[0:6]) - int(os.path.basename(comb_list[i][1])[0:6])) in [0, -1, -2])]
                for (s1_loc, s2_loc) in tqdm(comb_list, desc="[Load]"):
                    se_loc = s2_loc.replace("tif", "npy").replace("s2_", "se_")
                    self.samples.append({"s1": s1_loc, "s2": s2_loc, "se": se_loc, "id": os.path.basename(s2_loc)})


        # sort list of samples
        #if self.train_index:
        #    Tindex = np.load(self.train_index)
        #    self.samples = [self.samples[i] for i in Tindex]
        #    self.samples = sorted(self.samples, key=lambda i: i['id'])
        #else:
        #    self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, no_savanna=self.no_savanna,
                           igbp=False, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "/workplace/S2BYOL/OSCD"
    ds = OSCD_S2(data_dir, subset="train", use_s1=True, use_s2hr=True,
                 use_s2mr=True, no_savanna=True)
    s = ds.__getitem__(0)
    print("id:", s["id"], "\n",
          "input shape:", s["image"].shape, "\n")


