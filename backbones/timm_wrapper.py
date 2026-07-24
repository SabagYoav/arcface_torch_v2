import timm
import torch
import torch.nn as nn


class TimmBackboneWrapper(nn.Module):
    """Adapts a timm model (arbitrary native feature dim, e.g. Swin/MobileViT)
    to this repo's backbone convention: 112x112x3 input -> `num_features`-dim
    embedding, with the same autocast-then-fp32-head pattern used by IResNet
    (backbones/iresnet.py) and VisionTransformer (backbones/vit.py)."""

    def __init__(self, timm_name, num_features=512, fp16=False, img_size=112):
        super().__init__()
        self.fp16 = fp16
        self.trunk = timm.create_model(
            timm_name, pretrained=False, num_classes=0, in_chans=3, img_size=img_size,
        )
        feat_dim = self.trunk.num_features
        self.feature = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=num_features, bias=False),
            nn.BatchNorm1d(num_features=num_features, eps=2e-5),
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.trunk(x)
        x = self.feature(x.float() if self.fp16 else x)
        return x
