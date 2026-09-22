"""Identity interaction used by the frozen E2 late-fusion baseline."""

from __future__ import annotations

from typing import Any

from torch import nn

from ..registry import COMPONENTS
from .bi_hci import HCIOutput


@COMPONENTS.register("identity_interaction")
class IdentityInteraction(nn.Module):
    """Preserve both modalities exactly at a Pre-Stage interaction seam."""

    def forward(
        self,
        radar_features,
        radar_centers,
        radar_batch_index,
        vision_tokens,
        *,
        height: int,
        width: int,
        context,
        return_aux: bool = False,
        **_: Any,
    ) -> HCIOutput:
        aux = {"edge_count": 0} if return_aux else None
        return HCIOutput(radar_features, vision_tokens, aux)
