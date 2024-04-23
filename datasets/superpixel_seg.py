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

# indices of sentinel-2 high-/medium-/low-resolution bands
#S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_HR = [2, 3, 4]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]


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
    #s2 /= 10000  # normaliza 0~1
    # normniza -1 ~ 1
    # s2 = (s2 - 5000)/5000
    # s2 = s2.astype(np.float32)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    # normaliza 0~1
    #s1 /= 25
    #s1 += 1
    # normniza -1 ~ 1
    s1 = (s1 + 12.5)/12.5
    s1 += 1
    s1 = s1.astype(np.float32)
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

# this function for classification and most important is for weak supervised
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, superpixel=True,
                no_savanna=False, igbp=True, unlabeled=False, n_segments=100, sigma=2):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
        s2 = normalization(img)  # normaliza 0~1
        s2 = s2.astype(np.float32)
        #s2 = s2.swapaxes(2, 0) #这个错了,全错了
        s2 = np.rollaxis(s2, 0, 3)
        #segments = felzenszwalb(s2, scale=80, sigma=0.60, min_size=60)
        #segments = felzenszwalb(s2, scale=32, sigma=0.60, min_size=30)
        #segments = segments + 1 #为了和slic保持一致
        segments = slic(s2, n_segments=2000, sigma=1, start_label=1, multichannel=True)
        #print(segments.max())
        #print(sample["s2"].replace("tif", "npy").replace("s2_", "seb_"))
        #print(os.path.split(sample["s2"].replace("tif", "npy").replace("s2_", "se_"))[0])
        if not os.path.isdir(os.path.dirname(sample["s2"].replace("tif", "npy").replace("s2_", "ses_"))):
            os.makedirs(os.path.dirname(sample["s2"].replace("tif", "npy").replace("s2_", "ses_")))
        np.save(sample["s2"].replace("tif", "npy").replace("s2_", "ses_"), segments)

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


class DFC2020(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 subset="val",
                 no_savanna=False,
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        assert subset in ["val", "train", "test"]
        self.no_savanna = no_savanna

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(use_s2hr, use_s2mr, use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)

        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        # build list of sample paths
        if subset == "train":
            train_list = []
            #for seasonfolder in ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'brasilia',
            #                     'chongqing', 'cupertino', 'dubai', 'hongkong', 'lasvegas', 'milano', 'montpellier',
            #                     'mumbai', 'nantes', 'norcia', 'paris', 'pisa', 'rennes', 'rio', 'saclaye',
            #                     'saclayw', 'valencia']:
            #for seasonfolder in ['abudhabi',   'cupertino',      'L15-0387E-1276N','L15-0586E-1127N','L15-1014E-1375N','L15-1203E-1203N','L15-1298E-1322N','L15-1615E-1205N','L15-1691E-1211N','montpellier','saclaye',
                #'aguasclaras','dubai',          'L15-0434E-1218N','L15-0595E-1278N','L15-1015E-1062N','L15-1204E-1202N','L15-1335E-1166N','L15-1615E-1206N','L15-1703E-1219N','mumbai', 'saclayw',
                #'beihai',     'hongkong',       'L15-0457E-1135N','L15-0614E-0946N','L15-1025E-1366N','L15-1204E-1204N','L15-1389E-1284N','L15-1617E-1207N','L15-1709E-1112N','nantes', 'valencia',
                #'beirut',     'L15-0331E-1257N','L15-0487E-1246N','L15-0632E-0892N','L15-1049E-1370N','L15-1209E-1113N','L15-1438E-1134N','L15-1669E-1153N','L15-1716E-1211N','norcia',
                #'bercy',      'L15-0357E-1223N','L15-0506E-1204N','L15-0683E-1006N','L15-1138E-1216N','L15-1210E-1025N','L15-1439E-1134N','L15-1669E-1159N','L15-1748E-1247N','paris',
                #'bordeaux',   'L15-0358E-1220N','L15-0544E-1228N','L15-0760E-0887N','L15-1172E-1306N','L15-1276E-1107N','L15-1479E-1101N','L15-1669E-1160N','L15-1848E-0793N',
                #'brasilia',   'L15-0361E-1300N','L15-0566E-1185N','L15-0924E-1108N','L15-1185E-0935N','L15-1289E-1169N','L15-1481E-1119N','L15-1672E-1207N','lasvegas', 'rennes',
                #'chongqing',  'L15-0368E-1245N','L15-0577E-1243N','L15-0977E-1187N','L15-1200E-0847N','L15-1296E-1198N','L15-1538E-1163N','L15-1690E-1211N','milano', 'rio']:
                # for seasonfolder in ['abudhabi', ]:
            for seasonfolder in ['abudhabi', 'chongqing', 'L15-0361E-1300', 'L15-0487E-1246', 'L15-0586E-1127',
                                 'L15-0760E-0887', 'L15-1049E-1370', 'L15-1204E-1204', 'L15-1289E-1169',
                                 'L15-1479E-1101', 'L15-1630E-0988', 'L15-1690E-1211', 'L15-1848E-0793', 'paris',
                                 'aguasclaras', 'cupertino', 'L15-0368E-1245', 'L15-0506E-1204', 'L15-0595E-1278',
                                 'L15-0924E-1108', 'L15-1129E-0819', 'L15-1209E-1113', 'L15-1296E-1198',
                                 'L15-1481E-1119', 'L15-1666E-1189', 'L15-1691E-1211', 'lasvegas', 'pisa',
                                 'beihai', 'dubai', 'L15-0369E-1244', 'L15-0509E-1108', 'L15-0614E-0946',
                                 'L15-0977E-1187', 'L15-1138E-1216', 'L15-1210E-1025', 'L15-1298E-1322',
                                 'L15-1538E-1163', 'L15-1669E-1153', 'L15-1703E-1219', 'milano', 'rennes',
                                 'beirut', 'hongkong', 'L15-0387E-1276', 'L15-0544E-1228', 'L15-0632E-0892',
                                 'L15-1014E-1375', 'L15-1172E-1306', 'L15-1213E-1238', 'L15-1389E-1284',
                                 'L15-1546E-1154', 'L15-1669E-1160', 'L15-1709E-1112', 'montpellier', 'rio',
                                 'bercy', 'L15-0331E-1257', 'L15-0391E-1219', 'L15-0566E-1185', 'L15-0683E-1006',
                                 'L15-1015E-1062', 'L15-1185E-0935', 'L15-1249E-1167', 'L15-1438E-1134',
                                 'L15-1615E-1205', 'L15-1670E-1159', 'L15-1716E-1211', 'mumbai', 'saclaye',
                                 'bordeaux', 'L15-0357E-1223', 'L15-0434E-1218', 'L15-0571E-1302', 'L15-0697E-0874',
                                 'L15-1025E-1366', 'L15-1203E-1203', 'L15-1276E-1107', 'L15-1438E-1227',
                                 'L15-1615E-1206', 'L15-1672E-1207', 'L15-1748E-1247', 'nantes', 'saclayw',
                                 'brasilia', 'L15-0358E-1220', 'L15-0457E-1135', 'L15-0577E-1243', 'L15-0744E-0927',
                                 'L15-1031E-1300', 'L15-1204E-1202', 'L15-1281E-1035', 'L15-1439E-1134',
                                 'L15-1617E-1207', 'L15-1690E-1210', 'L15-1749E-1266', 'norcia', 'valencia']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder)) if "s2_" in x]
            train_list = [os.path.join(x, y) for x in train_list for y in os.listdir(os.path.join(path, x))]
            sample_dirs = train_list
            # path = os.path.join(path, "ROIs0000_validation", "s2_validation")
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                #s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                #lc_loc = s2_loc.replace("_s2_", "_dfc_").replace("s2_", "dfc_")
                self.samples.append({"s2": s2_loc, "id": os.path.basename(s2_loc)})
        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        return load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, no_savanna=self.no_savanna, igbp=False)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


if __name__ == "__main__":
    print("\n\nDFC2020 test")
    data_dir = '/workplace/OSCD_TS'
    ds = DFC2020(data_dir, subset="train", use_s1=False, use_s2hr=True, use_s2mr=False, use_s2lr=False, no_savanna=True)
    for i in range(len(ds)):
        s = ds.__getitem__(i)
        #print("id:", s["id"], "\n", "input shape:", s["image"].shape)
