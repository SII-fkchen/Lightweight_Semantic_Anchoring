# model.py
# Model definition: a lightweight dual-encoder (vision + text) that maps
# word images and multilingual word texts into a shared embedding space.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from transformers import DistilBertModel


class VisionEncoder(nn.Module):
    """Lightweight vision encoder: MobileNetV3-Small backbone + projection head."""

    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights)

        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        return F.normalize(x, p=2, dim=1)


class TextEncoder(nn.Module):
    """Lightweight text encoder: multilingual DistilBERT + projection head."""

    def __init__(self, embedding_dim=128, model_name='distilbert-base-multilingual-cased', freeze_layers=True):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name, use_safetensors=True)

        if freeze_layers:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            for layer in self.bert.transformer.layer[:3]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        x = self.projection(cls_token)
        return F.normalize(x, p=2, dim=1)


class DualEncoder(nn.Module):
    """Dual-encoder model combining the vision and text encoders."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.vision_encoder = VisionEncoder(embedding_dim, pretrained=False)
        self.text_encoder = TextEncoder(embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, input_ids, attention_mask):
        img_embeds = self.vision_encoder(images)
        text_embeds = self.text_encoder(input_ids, attention_mask)
        return img_embeds, text_embeds, self.logit_scale.exp()
