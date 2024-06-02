"""Sample evaluation script for track 1."""

import argparse
import importlib
from pathlib import Path

import torch
from torch import nn

from anomalib.data import MVTec
from anomalib.metrics import F1Max
from torchvision import transforms
from model_file.train import center_crop_arr
import numpy as np
from model_file.metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics
from model_file.test import MVTecDataset
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class MVTecDataset(Dataset):
    def __init__(self, data_path,transform=None, flag='train'):
        super(MVTecDataset, self).__init__()
        assert flag in ['train', 'test']
        self.root_path = data_path         # like: ./mvtec/bottle/test
        self.data_list = []
        self.flag = flag
        self.transform = transform
        _, self.cls_name = os.path.split(self.root_path)

        self.transform_mask = [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
        if self.flag == 'train':
            train_dir = os.path.join(self.root_path, self.flag, 'good')
            for file_name in os.listdir(train_dir):
                data = {}
                file_path = os.path.join(train_dir, file_name)
                data['file_path'] = file_path
                data['mask_path'] = ''
                data['label'] = 0
                data['defe_name'] = 'good'
                data['cls_name'] = self.cls_name
                data['flag'] = self.flag
                self.data_list.append(data)

        elif self.flag == 'test':
            test_dir = os.path.join(self.root_path, self.flag)
            for sub_cls in os.listdir(test_dir):
                sub_path = os.path.join(test_dir, sub_cls)
                for file_name in os.listdir(sub_path):
                    data = {}
                    file_path = os.path.join(sub_path, file_name)
                    data['file_path'] = file_path
                    # gt
                    if sub_cls != 'good':
                        mask_root_path = os.path.join(self.root_path, 'ground_truth')
                        file_pre_name = file_name.split('.')[0]
                        mask_path = os.path.join(mask_root_path, sub_cls, file_pre_name + '_mask.png')
                        assert os.path.exists(mask_path)
                        data['mask_path'] = mask_path
                        data['label'] = 1
                        data['defe_name'] = sub_cls
                    else:
                        data['mask_path'] = ""
                        data['label'] = 0       # only one class
                        data['defe_name'] = 'good'
                    data['cls_name'] = self.cls_name
                    data['flag'] = self.flag
                    self.data_list.append(data)

    def __getitem__(self, ind):
        item = self.data_list[ind]
        # image
        file_path = item['file_path']
        img = Image.open(file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        item['image'] = img
        # label
        label = torch.tensor(item['label'])
        item['label'] = label
        # mask
        if self.flag == 'test' and item['mask_path']:
            mask = Image.open(item['mask_path'])
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, 256, 256])
        item["gt"] = mask
        return item

    def __len__(self):
        return len(self.data_list)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str, category: str) -> nn.Module:
    """Load model.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.
        category (str): Category of the dataset.

    Note:
        We assume that the weight path contain the weights for all categories.
            For example, if the weight path is "/path/to/weights/", then the
            weights for each category should be stored as
            "/path/to/weights/bottle.pth", "/path/to/weights/zipper.pth", etc.


    Returns:
        nn.Module: Loaded model.
    """
    # get model class
    model_class = getattr(importlib.import_module(module_path), class_name)
    # instantiate model
    model = model_class()
    # load weights
    if weights_path:
        weight_file = Path(weights_path) / f"{category}.pth"
        model.load_state_dict(torch.load(weight_file))
    return model


def run(module_path: str, class_name: str, weights_path: str, dataset_path: str, category: str) -> None:
    """Run the evaluation script.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str | None, optional): Path to the model weights.
        dataset_path (str): Path to the dataset.
        category (str): Category of the dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset_path = os.path.join(dataset_path,category)
    # Instantiate and load the model
    model = load_model(module_path, class_name, weights_path, category)
    model.to(device)

    # Create the dataset
    # NOTE: We fix the image size to (256, 256) for consistent evaluation across all models.
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # datamodule = MVTec(root=dataset_path, eval_batch_size=1, image_size=(256, 256))
    # datamodule.setup()

    # Create the metrics
    image_metric = F1Max()
    pixel_metric = F1Max()
    dataset = MVTecDataset(dataset_path, transform=transform, flag='test')
    print(len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=int(1),
        shuffle=False,
        # sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    # Loop over the test set and compute the metrics
    for i, data in enumerate(loader):
        x = data['image']
        y = data['label']
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            output = model(data["image"].to(device))

            # Update the image metric
            image_metric.update(output["pred_score"].cpu(), data["label"])
            # Update the pixel metric
            pixel_metric.update(output["anomaly_map"].squeeze().cpu(), data["gt"].squeeze().cpu())
    image_score = image_metric.compute()
    print(image_score)
    pixel_score = pixel_metric.compute()
    print(pixel_score)


if __name__ == "__main__":
    args = parse_args()
    run(args.model_path, args.model_name, args.weights_path, args.dataset_path, args.category)
