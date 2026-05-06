# Copyright (c) ModelScope Contributors. All rights reserved.

from typing import Optional

import torch

from .base import ConfigLossScale
from .utils import calculate_loss_scale, med_calculate_loss_scale, med_calculate_loss_scale_v2


# class AgentFlanLossScale(ConfigLossScale):
#     is_binary = False
#     loss_scale_config = 'agentflan.json'

#     def get_loss_scale(self, context: str, *, query: Optional[str] = None):
#         if isinstance(context, str):
#             return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
#         return super().get_loss_scale(context)

# def med_calculate_loss_scale(query: str,
#                          response: str,
#                          response_loss_scale_map: Dict[str, list], coef1: float, coef2: float,
#                          query_loss_scale_map: Optional[Dict[str, list]] = None) -> Tuple[List[str], List[float]]:


class MedLossScale(ConfigLossScale):
    is_binary = False

    loss_scale_config = "repsv_train_lingshu7b_correction.json"
    coef1 = 1.0
    coef2 = 2.5
    coef2_reset_ratio: float = 0.0  # ~ratio% of coef2 weights randomly reset to 1.0 each epoch

    # def apply_epoch_random_reset(self, loss_scale: torch.Tensor, epoch: float,
    #                              global_step: int) -> torch.Tensor:
    #     """Randomly reset ~coef2_reset_ratio of coef2 positions to 1.0 per epoch."""
    #     mask = (torch.abs(loss_scale - self.coef2) < 1e-2)
    #     if mask.sum() == 0:
    #         return loss_scale
    #     g = torch.Generator(device=loss_scale.device).manual_seed(
    #         int(epoch * 1e9) + global_step)
    #     reset = torch.rand(mask.sum(), device=loss_scale.device,
    #                       generator=g) < self.coef2_reset_ratio
    #     out = loss_scale.clone()
    #     out[mask] = torch.where(reset, torch.ones_like(out[mask]), out[mask])
    #     return out

    def get_loss_scale(self,
                       context: str, *, query: Optional[str] = None):
        if isinstance(context, str):
            return med_calculate_loss_scale_v2(query=query, response=context, response_loss_scale_map=self.loss_scale_map, coef1=self.coef1, coef2=self.coef2)
        return super().get_loss_scale(context)