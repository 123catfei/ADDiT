import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image

class MvtecDataset(Dataset):

    def __init__(self,args,flag='train'):
        assert flag in ['train', 'test', 'valid']
        self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
        self.data_dir = "./mvtec/bottle/train/good"
        self.load_data()
    
    def load_data(self):
        data_list = os.listdir(self.data_dir)
        self.data_paths = [os.path.join(self.data_dir,data_name) for data_name in data_list]
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        data = Image.open(data_path).convert('RGB')
        data = self.transform(data)
        return data,0
    
    def __len__(self):
        return len(self.data_paths)
    


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
