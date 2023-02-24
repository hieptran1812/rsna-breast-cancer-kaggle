import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")
    
    
class RsnaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = len(cfg.classes)
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained,
                num_classes=0, global_pool='', in_chans=self.cfg.in_channels)
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        self.global_pool = GeM(p_trainable=False)
        backbone_out = 1280
        self.head = torch.nn.Linear(backbone_out, self.n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()
   
    def forward(self, batch):
        x = batch['input']
        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        
        logits = self.head(x)
        outputs = {}
        if self.training:
            loss = self.loss_fn(logits, batch['target'].float())
            outputs['loss'] = loss
        else:
            outputs['logits'] = logits
        return outputs
