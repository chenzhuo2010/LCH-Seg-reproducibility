from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class TinyComponentDetectionLoss(torch.nn.Module):
    """Primary LCH-Seg loss: DiceCE plus a local component-hit term."""

    def __init__(
        self,
        base_loss: torch.nn.Module,
        component_weight: float = 0.25,
        tiny_max_size: int = 10,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.component_weight = component_weight
        self.tiny_max_size = tiny_max_size
        self.eps = eps

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(net_output, target)
        target_lbl = target[:, 0].long() if target.ndim == net_output.ndim else target.long()
        probability = torch.softmax(net_output, dim=1)[:, 1]
        component_loss = self._small_component_loss(probability, target_lbl)
        return base + self.component_weight * component_loss

    def _small_component_loss(
        self,
        probability: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for batch_index in range(target_label.shape[0]):
            mask = target_label[batch_index].detach().cpu().numpy() > 0
            if not mask.any():
                continue
            structure = np.ones((3,) * mask.ndim, dtype=np.uint8)
            labels, component_count = ndimage.label(mask, structure=structure)
            if component_count == 0:
                continue
            sizes = np.bincount(labels.ravel())
            component_ids = np.where((sizes >= 1) & (sizes <= self.tiny_max_size))[0]
            for component_id in component_ids[component_ids > 0]:
                component = torch.as_tensor(
                    labels == int(component_id),
                    device=probability.device,
                    dtype=torch.bool,
                )
                values = torch.clamp(
                    probability[batch_index][component],
                    self.eps,
                    1.0 - self.eps,
                )
                hit_probability = 1.0 - torch.prod(1.0 - values)
                losses.append(-torch.log(torch.clamp(hit_probability, min=self.eps)))
        return torch.stack(losses).mean() if losses else probability.sum() * 0.0


class nnUNetTrainerTinyComponentClean1000epochs(nnUNetTrainer):
    """nnU-Net trainer used for the primary matched OOF LCH-Seg checkpoint."""

    def _build_loss(self):
        if self.label_manager.has_regions:
            return super()._build_loss()
        base = DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        loss = TinyComponentDetectionLoss(
            base,
            component_weight=0.25,
            tiny_max_size=10,
        )
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**index) for index in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
