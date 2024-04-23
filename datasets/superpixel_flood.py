import os
import re
import glob
import rasterio
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from skimage.segmentation import felzenszwalb, slic, mark_boundaries

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


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

#L8_BANDS_HR = [2, 3, 4, 5]
L8_BANDS_HR = [2, 3, 4]
L8_BANDS_MR = [5, 6, 7, 9, 12, 13]
L8_BANDS_LR = [1, 10, 11]

L1_BANDS = [1, 2, 1]


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read(L1_BANDS)
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    # normaliza 0~1
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    return s1


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
    return l8


# this function for classification and most important is for weak supervised
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, superpixel=True,
                no_savanna=False, igbp=True, unlabeled=False, n_segments=100, sigma=2):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        #img = load_l8(sample["l8"], use_s2hr, use_s2mr, use_s2lr)
        img = load_s1(sample["s1"])
        s2 = normalization(img)  # normaliza 0~1
        s2 = s2.astype(np.float32)
        s2 = np.rollaxis(s2, 0, 3)
        segments = slic(s2, n_segments=1000, sigma=1, start_label=1, multichannel=True)
        print(segments.max())
        #if not os.path.isdir(os.path.dirname(sample["l8"].replace("tif", "npy").replace("l8_", "sl8_"))):
        #    os.makedirs(os.path.dirname(sample["l8"].replace("tif", "npy").replace("l8_", "sl8_")))
        #np.save(sample["l8"].replace("tif", "npy").replace("l8_", "sl8_"), segments)
        if not os.path.isdir(os.path.dirname(sample["l8"].replace("tif", "npy").replace("l8_", "ss1_"))):
            os.makedirs(os.path.dirname(sample["l8"].replace("tif", "npy").replace("l8_", "ss1_")))
        np.save(sample["l8"].replace("tif", "npy").replace("l8_", "ss1_"), segments)

    # segmentate the image to superpixels
    if superpixel:
        segments = None
    else:
        segments = None

    # load label
    if unlabeled:
        return {'image': img, 'segments': segments, 'id': sample["id"]}
    else:
        return {'image': img, 'segments': segments, 'id': sample["id"]}


class OSCD(data.Dataset):
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
        super(OSCD, self).__init__()

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
        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)

        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        if subset == "train":
            train_list = []
            for seasonfolder in ['california', ]:
            #for seasonfolder in ['beirut', ]:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder)) if "s1_" in x]
            train_list = [os.path.join(x, y) for x in train_list for y in os.listdir(os.path.join(path, x))]
            sample_dirs = train_list
            #path = os.path.join(path, "ROIs0000_validation", "s2_validation")
        else:
            path = os.path.join(path, "ROIs0000_test", "s1_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for (s2_loc) in tqdm(s2_locations, desc="[Load]"):
                l8_loc = s2_loc.replace("_s1_", "_l8_").replace("s1_", "l8_")
                if os.path.isfile(l8_loc):
                    self.samples.append({"l8": l8_loc, "s1": s2_loc, "id": os.path.basename(s2_loc)})
                else:
                    continue
        # sort list of samples
        if self.train_index:
            Tindex = np.load(self.train_index)
            self.samples = [self.samples[i] for i in Tindex]
            self.samples = sorted(self.samples, key=lambda i: i['id'])
        else:
            self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, no_savanna=self.no_savanna,
                           igbp=False, unlabeled=self.unlabeled)
        return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


if __name__ == "__main__":
    print("\n\nDFC2020 test")
    data_dir = '/workplace/Floods'
    ds = OSCD(data_dir, subset="train", use_s1=False, use_s2hr=True, use_s2mr=False, use_s2lr=False, no_savanna=True)
    for i in range(len(ds)):
        s = ds.__getitem__(i)
        #print("id:", s["id"], "\n", "input shape:", s["image"].shape)
