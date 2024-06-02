import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from .modules import Block, _cfg, PatchEmbed, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from sklearn.metrics.pairwise import cosine_similarity

from einops import rearrange
import os


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'vit_base_patch16_224_8k_vocab',
    'vit_large_patch16_224_8k_vocab',
]


class PatchLevelReconstructionModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 outplane=272, feature_size=(14, 14), pos_embed_type='learned', idden_dim=256, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.input_proj = nn.Linear(outplane, embed_dim)
        self.output_proj = nn.Linear(embed_dim, outplane)
        self.feature_size = feature_size
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)
        self.criterion_mse = nn.MSELoss()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, input, tokenimg_fe, batch_size, seq_len=196):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> b (h w) c "
        )  # (H x W) x B x C
        feature_tokens = self.input_proj(feature_tokens)  # B x (H x W) x C

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        feature_tokens = torch.cat((cls_tokens, feature_tokens), dim=1)

        # mask_token = self.mask_token.expand(batch_size, seq_len, -1)
        # replace the masked visual tokens by mask_token
        # w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        # feature_tokens = feature_tokens * (1 - w) + mask_token * w
        # pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C

        # output_decoder, _ = self.transformer(feature_tokens, pos_embed)  # (H x W) x B x C

        if self.pos_embed is not None:
            feature_tokens = feature_tokens + self.pos_embed
        feature_tokens = self.pos_drop(feature_tokens)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            feature_tokens = blk(feature_tokens, rel_pos_bias=rel_pos_bias)  # (1,197,768)

        feature_rec_tokens = self.output_proj(feature_tokens)  # (H x W) x B x C  (30,197,272)
        feature_rec_tokens = feature_rec_tokens[:, 1:]  # (N, 196, 272)

        feature_rec = rearrange(
            feature_rec_tokens, "b (h w) c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W  (30,272,14,14)
        clsnames = input["clsname"]
        filenames = input["filename"]
        clsname = clsnames
        # feat_rec = feature_rec.mean(dim=0)
        feat_rec = feature_rec[9]
        for filename in filenames:
            filedir, filename = os.path.split(filename)
            _, defename = os.path.split(filedir)
            filename_, _ = os.path.splitext(filename)
            save_dir = os.path.join("result_recon", clsname, defename)
            os.makedirs(save_dir, exist_ok=True)
            feature_rec_np = feat_rec.detach().cpu().numpy()
            np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - tokenimg_fe["feature_align"]) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W (N,1,14,14)  N=30

        # loss
        patches_distribution = torch.t(pred.reshape(batch_size, 196))  # (30,196)->(196,30)
        samples_uniform = torch.full((batch_size, 196), 1.0 / batch_size, dtype=torch.float32).to(patches_distribution)

        RMSE_patches_score = torch.mean(patches_distribution, dim=1, keepdim=True)  # (196,1)
        RMSE_patches_loss = torch.mean(RMSE_patches_score)
        RMSE_patches_score_map = RMSE_patches_score.reshape(1, 1, 14, 14)
        RMSE_patches_score_map = self.upsample(RMSE_patches_score_map)  # B x 1 x H x W  (30,1,224,224)

        MSELoss = self.criterion_mse(feature_rec, tokenimg_fe["feature_align"])  # (1,)
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "patches_distribution": patches_distribution,
            "RMSE_patches_score": RMSE_patches_score,
            "RMSE_patches_score_map": RMSE_patches_score_map,
            "MSELoss": MSELoss,
            "RMSE_patches_loss": RMSE_patches_loss
        }

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    # def forward(self, x, bool_masked_pos, return_all_tokens=False):
    #     """
    #         x : (N, 3, 224, 224)
    #         bool_masked_pos: (N, 196)
    #     """
    #     x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
    #     x = x[:, 1:]  # (N, 196, 768)
    #     if return_all_tokens:  # False
    #         return self.lm_head(x)
    #     else:
    #         # return the masked tokens
    #         return self.lm_head(x[bool_masked_pos])

    def unpatchify(self, x):
        """
            x: (N, L, patch_size**2 *3)
            imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs


@register_model
def vit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = PatchLevelReconstructionModel(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def vit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    model = PatchLevelReconstructionModel(
        patch_size=16, embed_dim=256, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
