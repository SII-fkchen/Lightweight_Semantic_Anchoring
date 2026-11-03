# loss.py
# Loss definition: CLIP-style image-text contrastive loss plus a semantic-ID
# invariance loss that clusters samples of the same word ID across modalities
# and languages.

import torch
import torch.nn as nn


class InvarianceLoss(nn.Module):
    """
    Combined contrastive + invariance loss.

    The contrastive term aligns image and text embeddings using symmetric
    image-to-text and text-to-image cross entropy over the similarity matrix.
    The invariance term pulls together all embeddings that share the same
    semantic word ID, regardless of modality or language.

    Args:
        lambda_inv: weight of the semantic invariance loss.
    """

    def __init__(self, lambda_inv=0.5):
        super().__init__()

        self.lambda_inv = lambda_inv
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        img_embeds,
        text_embeds,
        labels,
        logit_scale=None
    ):
        """
        Args:
            img_embeds:
                [Batch, Dim] normalized image embeddings.

            text_embeds:
                [Batch, Dim] normalized text embeddings.

            labels:
                [Batch] semantic word IDs.

            logit_scale:
                Learnable temperature scaling factor from the dual encoder.
                If None, a fixed CLIP-style scale is used for compatibility.
        """

        batch_size = img_embeds.shape[0]

        # ------------------------------------------------------------
        # Symmetric image-text contrastive alignment (L_ITC)
        # ------------------------------------------------------------
        if logit_scale is None:
            scale = 100.0
        else:
            scale = logit_scale

        logits_per_image = (
            torch.matmul(img_embeds, text_embeds.T)
            * scale
        )

        logits_per_text = logits_per_image.T

        labels_gt = torch.arange(
            batch_size,
            device=img_embeds.device
        )

        loss_i2t = self.ce_loss(
            logits_per_image,
            labels_gt
        )

        loss_t2i = self.ce_loss(
            logits_per_text,
            labels_gt
        )

        loss_contrastive = (
            loss_i2t + loss_t2i
        ) / 2


        # ------------------------------------------------------------
        # Semantic invariance alignment (L_INV)
        # ------------------------------------------------------------
        labels = labels.to(img_embeds.device)

        all_embeds = torch.cat(
            [
                img_embeds,
                text_embeds
            ],
            dim=0
        )

        all_labels = torch.cat(
            [
                labels,
                labels
            ],
            dim=0
        )

        sim_matrix = torch.matmul(
            all_embeds,
            all_embeds.T
        )

        pos_mask = (
            all_labels.unsqueeze(0)
            ==
            all_labels.unsqueeze(1)
        ).float()

        # Remove self similarity
        pos_mask.fill_diagonal_(0)


        if pos_mask.sum() > 0:
            sim_pos = (
                sim_matrix * pos_mask
            ).sum() / pos_mask.sum()

            loss_invariance = 1.0 - sim_pos

        else:
            loss_invariance = torch.tensor(
                0.0,
                device=img_embeds.device
            )


        # ------------------------------------------------------------
        # Total objective:
        # L = L_ITC + lambda * L_INV
        # ------------------------------------------------------------
        loss = (
            loss_contrastive
            +
            self.lambda_inv * loss_invariance
        )

        return loss