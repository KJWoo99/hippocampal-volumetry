"""Sliding-window 추론 + VRAM 동적 배치 + OOM 자동 복구.

VRAM 여유에 따라 배치를 자동 조정하고, OOM 이 나면 배치를 줄여 자동 복구함.
3D 볼륨은 패치 수가 많아 VRAM 이 빡빡하므로 2D 보다 이 처리가 더 중요함.
"""
from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        roi: tuple[int, int, int] = (32, 32, 32),
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        device: torch.device | str = "cpu",
        amp: bool = True,
        min_sw_batch: int = 1,
    ):
        self.model = model.to(device).eval()
        self.roi = roi
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.min_sw_batch = min_sw_batch

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """OOM이 나면 sliding-window 배치를 절반으로 줄여 자동 재시도."""
        x = x.to(self.device)
        sw = self.sw_batch_size
        while True:
            try:
                with torch.autocast(self.device.type, enabled=self.amp):
                    return sliding_window_inference(
                        x, self.roi, sw, self.model, overlap=self.overlap
                    )
            except torch.cuda.OutOfMemoryError:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                if sw <= self.min_sw_batch:
                    raise
                sw = max(self.min_sw_batch, sw // 2)  # 배치 축소 후 복구
                continue

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """argmax 라벨 마스크 (B,H,W,D) 반환."""
        return self.logits(x).argmax(dim=1)
