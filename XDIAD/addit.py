import torch
from torch import nn
from model_file.models import DiT_models
from diffusers.models import AutoencoderKL
from model_file.diffusion import create_diffusion
from torchvision import transforms
from model_file.train import center_crop_arr
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from easydict import EasyDict
from model_file.model_helper import ModelHelper
from model_file.misc_helper import update_config
import yaml
from scipy.ndimage import gaussian_filter
from torchvision.utils import save_image
import numpy as np

class ADDiT(nn.Module):
    def __init__(self):
        super(ADDiT, self).__init__()
        self.image_size = 256
        latent_size = self.image_size // 8
        self.diff = DiT_models["DiT-S/2"](
                input_size=latent_size,
                num_classes=1
        )
        self.diffusion = create_diffusion(str(1000))
        self.vae = AutoencoderKL.from_pretrained(f"model_file/stabilityai/sd-vae-ft-ema")
        self.get_feature_model()
        self.device = None
        self.save_file_path = "model_file/sample.png"
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        

    def get_feature_model(self):
        with open("./model_file/config.yaml") as f:
            config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        config = update_config(config)
        self.effmodel = ModelHelper(config.net)
        
    def reconstruction(self,x):
        x = x.to(self.device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        t = torch.tensor([100], device=self.device)
        y = torch.tensor([0]).to(self.device)
        model_kwargs = dict(y=y, cfg_scale=4.0)

        # diffusion forward
        x_t = self.diffusion._q_sample(x, t)
        z = x_t
        # Setup classifier-free guidance:
        class_labels = [0]
        n = len(class_labels)
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1] * n, device=self.device)
        y = torch.cat([y, y_null], 0)
        # Sample images:
        self.diffusion.num_timesteps = 100
        samples = self.diffusion.p_sample_loop(
            self.diff.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=self.device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = self.vae.decode(samples / 0.18215).sample
        save_image(samples, self.save_file_path, nrow=4, normalize=True, value_range=(-1, 1))

    
    def create_map(self,test_data, rec_data):
        torch.manual_seed(0)
        torch.set_grad_enabled(False)
        input1 = {"image": test_data}
        tokenimg_te = self.effmodel(input1)
        input2 = {"image": rec_data}
        tokenimg_rec = self.effmodel(input2)
        pred = torch.sqrt(
            torch.sum((tokenimg_te["feature_align"] - tokenimg_rec["feature_align"]) ** 2, dim=1, keepdim=True)
        ) #[1,1,16,16]
        # print(pred.shape)
        unsample = nn.UpsamplingBilinear2d(scale_factor=16)
        score_map = unsample(pred) #[1,1,256,256]
        # print(score_map.shape)
        # exit(0)
        score_map = score_map.squeeze()#[256,256]
        score_map = score_map.cpu().numpy()
        score_map = gaussian_filter(score_map, sigma=4)
        pixel_scores = np.asarray(score_map)
        pixel_scores = pixel_scores[np.newaxis, :]
        img_scores = np.max(pixel_scores.reshape(pixel_scores.shape[0], -1), axis=-1)
        pixel_scores = torch.tensor(pixel_scores)
        img_scores = torch.tensor(img_scores)
        output = {}
        output["pred_score"] = img_scores
        output["anomaly_map"] = pixel_scores
        return output

    def forward(self, batch):
        self.device = next(self.diff.parameters()).device
        self.reconstruction(batch)
        recon_img = Image.open(self.save_file_path).convert('RGB')
        if self.transform is not None:
            recon_img = self.transform(recon_img)
        recon_img = recon_img.unsqueeze(0)
        recon_img = recon_img.to(self.device)
        return self.create_map(batch,recon_img)
        