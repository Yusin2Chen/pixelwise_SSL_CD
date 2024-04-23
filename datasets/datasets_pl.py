import os
import glob
import rasterio
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import torch.utils.data as data

def resiz_4pl(img, size):
    imgs = np.zeros((img.shape[0], size[0], size[1]))
    for i in range(img.shape[0]):
        per_img = np.squeeze(img[i, :, :])
        per_img = cv2.resize(per_img, size, interpolation=cv2.INTER_NEAREST)
        imgs[i, :, :] = per_img
    return imgs

# standar
pl_MEAN = np.array([620.56866, 902.1002, 1011.31476, 2574.5764])
pl_STD  = np.array([219.36754, 254.2806, 350.12357, 535.43195])


def normalize_pl(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - pl_MEAN[i]) / pl_STD[i]
    return imgs

# indices of sentinel-2 high-/medium-/low-resolution bands
pl_BANDS_HR = [1, 2, 3, 4]

# util function for reading s2 data
def load_pl(path):

    with rasterio.open(path) as data:
        pl = data.read(pl_BANDS_HR)
    pl = pl.astype(np.float32)
    pl = resiz_4pl(pl, (256, 256))
    pl = pl.astype(np.float32)
    pl = np.clip(pl, 0, 10000)
    pl = normalize_pl(pl)
    return pl


def load_lc(path):
    # load labels
    with rasterio.open(path) as data:
        lc = data.read()
        lc = np.argmax(lc, axis=0)
        #lc = lc.astype(np.float32)
        #lc = resiz_4pl(lc[np.newaxis, :, :], (256, 256))
        lc = lc.astype(np.int)
        lc = lc.squeeze()

    return lc


# util function for reading data from single sample
def load_sample(sample):

    img = load_pl(sample["as2"])
    img = np.concatenate((img, load_pl(sample["bs2"])), axis=0)

    alc = load_lc(sample["alc"])
    blc = load_lc(sample["blc"])

    return {'image': img, 'alc': alc, 'blc': blc, 'id': sample["id"]}



class MUSD_PL(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 subset="val"):
        """Initialize the dataset"""

        # inizialize
        super(MUSD_PL, self).__init__()
        assert subset in ["val", "train", "test"]
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
                                                '4768_4131_13',
                '1973_3709_13', '2850_4139_13', '4791_3920_13', '6204_3495_13', '7026_3201_13',
                '2006_3280_13',
                '2029_3764_13', '3002_4273_13', '4806_3588_13', '6353_3661_13', '7312_3008_13',
                '2065_3647_13', '3830_3914_13', '4838_3506_13', '6381_3681_13', '7367_5050_13',
                '2196_3885_13', '3998_3016_13', '4856_4087_13', '6466_3380_13', '7513_4968_13',
                                '4062_3943_13',                 '6468_3360_13',
                '2235_3403_13', '4127_2991_13', '4881_3344_13', '6475_3361_13', '7517_4908_13',
                                '4169_3944_13',
                '2415_3082_13', '4223_3246_13', '5111_4560_13', '6678_3579_13', '8077_5007_13',
                                '4240_3972_13',
                '2459_4406_13', '4254_2915_13', '5125_4049_13', '6688_3456_13',
                                '4397_4302_13',
                '2470_5030_13', '4421_3800_13', '5195_3388_13', '6730_3430_13']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder)) if "Images" in x]
            sample_dirs = train_list
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/201*.tif"), recursive=True)
            s2_locations = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]))
            as2_loc = s2_locations[0]
            bs2_loc = s2_locations[-1]
            alc_loc = as2_loc.replace("Images", "Labels")
            blc_loc = bs2_loc.replace("Images", "Labels")
            if os.path.exists(alc_loc):
                self.samples.append({"as2": as2_loc, "bs2": bs2_loc, "alc": alc_loc, "blc": blc_loc, "id": folder.split('/')[0]})

        print("loaded", len(self.samples), "samples from the dfc2020 subset", subset)


    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_sample(sample)
        return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "/workplace/S2BYOL/OSCD"
    ds = MUSD_PL(data_dir, subset="train")
    s = ds.__getitem__(0)
    print("id:", s["id"], "\n",
          "input shape:", s["image"].shape, "\n")


