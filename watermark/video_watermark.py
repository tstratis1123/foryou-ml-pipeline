"""VideoSeal neural watermarking for video content."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default batch size for frame processing to balance GPU memory and throughput.
_DEFAULT_BATCH_SIZE = 16

# Strength factor controlling the magnitude of the watermark residual.
_DEFAULT_STRENGTH = 0.03


# ---------------------------------------------------------------------------
# Neural network components
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WatermarkEncoder(nn.Module):
    """U-Net-style encoder that produces a residual from (frame, payload).

    Architecture overview:
        1. A small MLP expands the payload vector into a spatial feature map
           that is concatenated with the bottleneck of the U-Net.
        2. The encoder path downsamples the input frame through convolution
           blocks.
        3. The decoder path upsamples and concatenates skip connections to
           produce a per-pixel residual with 3 output channels (RGB).
    """

    def __init__(self, payload_length: int) -> None:
        super().__init__()
        self.payload_length = payload_length

        # ----- Encoder path -----
        self.enc1 = _ConvBlock(3, 64)
        self.enc2 = _ConvBlock(64, 128)
        self.enc3 = _ConvBlock(128, 256)

        # ----- Payload projection -----
        # The payload is projected to a spatial feature map that is tiled to
        # match the bottleneck spatial dimensions and concatenated channel-wise.
        self.payload_fc = nn.Sequential(
            nn.Linear(payload_length, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ----- Decoder path (256 + 256 from payload → up to 3 channels) -----
        self.dec3 = _ConvBlock(512, 256)  # bottleneck + payload
        self.dec2 = _ConvBlock(256 + 128, 128)  # skip from enc2
        self.dec1 = _ConvBlock(128 + 64, 64)  # skip from enc1

        # Final 1x1 conv producing the residual. Uses Tanh to bound output.
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh(),
        )

        self.pool = nn.MaxPool2d(2)

    def forward(self, frame: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        """Produce a residual image for *frame* carrying *payload*.

        Args:
            frame: (B, 3, H, W) input frame tensor normalised to [0, 1].
            payload: (B, payload_length) binary tensor.

        Returns:
            Residual tensor of shape (B, 3, H, W) in roughly [-1, 1].
        """
        # Encoder
        e1 = self.enc1(frame)  # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)

        # Payload → spatial feature map
        p = self.payload_fc(payload)  # (B, 256)
        p = p.unsqueeze(-1).unsqueeze(-1)  # (B, 256, 1, 1)
        p = p.expand(-1, -1, e3.shape[2], e3.shape[3])  # tile spatially

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([e3, p], dim=1))  # (B, 256, H/4, W/4)
        d3_up = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, e2], dim=1))  # (B, 128, H/2, W/2)
        d2_up = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, e1], dim=1))  # (B, 64, H, W)

        residual = self.out_conv(d1)  # (B, 3, H, W)
        return residual


class WatermarkDecoder(nn.Module):
    """CNN decoder that recovers the payload from a watermarked frame.

    Architecture: conv blocks → global average pooling → FC → sigmoid.
    """

    def __init__(self, payload_length: int) -> None:
        super().__init__()
        self.payload_length = payload_length

        self.features = nn.Sequential(
            _ConvBlock(3, 64),
            nn.MaxPool2d(2),
            _ConvBlock(64, 128),
            nn.MaxPool2d(2),
            _ConvBlock(128, 256),
            nn.MaxPool2d(2),
            _ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, payload_length),
            nn.Sigmoid(),
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """Predict payload bits from a (possibly watermarked) frame.

        Args:
            frame: (B, 3, H, W) tensor normalised to [0, 1].

        Returns:
            (B, payload_length) tensor of bit probabilities in [0, 1].
        """
        x = self.features(frame)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VideoSealWatermark:
    """GPU-accelerated neural watermark for video content.

    Uses a learned encoder-decoder architecture (VideoSeal) to embed a
    binary message into video frames in a way that is:
      - Imperceptible to viewers.
      - Robust to re-encoding, cropping, and scaling.
      - Extractable without the original video (blind extraction).

    The encoder network takes each frame (or batch of frames) and the
    payload, and produces a residual that is added to the frame.  The
    decoder network recovers the payload from the watermarked frames.
    """

    def __init__(
        self,
        model_path: str,
        payload_length: int = 64,
        device: str = "cuda",
    ) -> None:
        """Initialise the VideoSeal watermarker.

        Args:
            model_path: Path to the pre-trained VideoSeal model weights.
                The file should be a dictionary with keys ``"encoder"`` and
                ``"decoder"`` mapping to the respective ``state_dict``.
                If the file does not exist the networks are initialised
                randomly (useful for training from scratch).
            payload_length: Number of bits in the watermark payload.
            device: Torch device string (e.g. ``"cuda"`` or ``"cpu"``).
        """
        self.model_path = model_path
        self.payload_length = payload_length
        self.device = device
        self.strength: float = _DEFAULT_STRENGTH
        self.batch_size: int = _DEFAULT_BATCH_SIZE

        # Instantiate networks.
        self.encoder = WatermarkEncoder(payload_length)
        self.decoder = WatermarkDecoder(payload_length)

        # Load pre-trained weights if available.
        if os.path.isfile(model_path):
            logger.info("Loading model weights from %s", model_path)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
        else:
            logger.warning(
                "Model weights not found at %s — using random initialisation.",
                model_path,
            )

        # Move to device and set eval mode.
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _payload_to_tensor(payload: bytes, payload_length: int, device: str) -> torch.Tensor:
        """Convert a byte sequence to a float tensor of 0.0 / 1.0 bits.

        If the payload contains fewer bits than *payload_length* the result is
        zero-padded.  If it contains more bits the excess is silently truncated.

        Returns:
            Tensor of shape ``(payload_length,)`` on *device*.
        """
        bits: list[float] = []
        for byte_val in payload:
            for shift in range(7, -1, -1):
                bits.append(float((byte_val >> shift) & 1))
        # Truncate or pad.
        bits = bits[:payload_length]
        bits += [0.0] * (payload_length - len(bits))
        return torch.tensor(bits, dtype=torch.float32, device=device)

    @staticmethod
    def _tensor_to_bytes(bits_tensor: torch.Tensor) -> bytes:
        """Convert a binary tensor (values rounded to 0/1) back to bytes."""
        bits = (bits_tensor > 0.5).int().cpu().tolist()
        # Pad to a multiple of 8.
        bits += [0] * ((8 - len(bits) % 8) % 8)
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte_val = 0
            for bit in bits[i : i + 8]:
                byte_val = (byte_val << 1) | bit
            result.append(byte_val)
        return bytes(result)

    @staticmethod
    def _read_video_frames(video_path: str) -> tuple[list[NDArray[np.uint8]], float, int, int]:
        """Decode all frames from a video file using OpenCV.

        Returns:
            (frames, fps, width, height) where each frame is an HxWx3 BGR
            uint8 array.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If OpenCV cannot open the video.
        """
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[NDArray[np.uint8]] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from video: {video_path}")

        logger.info(
            "Read %d frames from %s (%.1f fps, %dx%d)",
            len(frames),
            video_path,
            fps,
            width,
            height,
        )
        return frames, fps, width, height

    @staticmethod
    def _write_video_frames(
        frames: list[NDArray[np.uint8]],
        output_path: str,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Encode frames to a video file using OpenCV's VideoWriter.

        Uses the ``mp4v`` FourCC codec and writes to *output_path*.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV VideoWriter failed to open: {output_path}")
        for frame in frames:
            writer.write(frame)
        writer.release()
        logger.info("Wrote %d frames to %s", len(frames), output_path)

    def _frames_to_tensor(self, frames: list[NDArray[np.uint8]]) -> torch.Tensor:
        """Convert a list of BGR uint8 frames to a (N, 3, H, W) float tensor.

        Pixel values are normalised to [0, 1] and the channel order is
        converted from BGR (OpenCV) to RGB.
        """
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        # Stack to (N, H, W, 3), then permute to (N, 3, H, W).
        arr = np.stack(rgb_frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return tensor

    @staticmethod
    def _tensor_to_frames(tensor: torch.Tensor) -> list[NDArray[np.uint8]]:
        """Convert a (N, 3, H, W) float [0,1] tensor back to BGR uint8 frames."""
        arr = tensor.clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0).byte().cpu().numpy()
        frames: list[NDArray[np.uint8]] = []
        for i in range(arr.shape[0]):
            bgr = cv2.cvtColor(arr[i], cv2.COLOR_RGB2BGR)
            frames.append(bgr)
        return frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, video_path: str, payload: bytes, output_path: str) -> None:
        """Embed a neural watermark into a video.

        Args:
            video_path: Path to the source video file.
            payload: Binary payload to embed (must not exceed
                ``payload_length`` bits).
            output_path: Path where the watermarked video will be saved.

        Raises:
            FileNotFoundError: If the source video does not exist.
            RuntimeError: If video decoding/encoding fails.
        """
        logger.info(
            "Embedding video watermark: payload=%d bytes, video=%s",
            len(payload),
            video_path,
        )

        # 1. Decode video into frames.
        frames, fps, width, height = self._read_video_frames(video_path)

        # 2. Prepare payload tensor (replicated for each frame in batch).
        payload_tensor = self._payload_to_tensor(payload, self.payload_length, self.device)

        # 3. Process frames in batches through the encoder.
        watermarked_frames: list[NDArray[np.uint8]] = []
        with torch.no_grad():
            for start in range(0, len(frames), self.batch_size):
                batch_frames = frames[start : start + self.batch_size]
                batch_tensor = self._frames_to_tensor(batch_frames)
                batch_size = batch_tensor.shape[0]

                # Expand payload to match batch size: (B, payload_length).
                payload_batch = payload_tensor.unsqueeze(0).expand(batch_size, -1)

                # Encoder produces a residual.
                residual = self.encoder(batch_tensor, payload_batch)

                # Add scaled residual to the original frames.
                watermarked = batch_tensor + self.strength * residual
                watermarked_frames.extend(self._tensor_to_frames(watermarked))

        # 4. Re-encode to video.
        self._write_video_frames(watermarked_frames, output_path, fps, width, height)

        logger.info("Watermarked video saved to %s", output_path)

    def extract(self, watermarked_path: str) -> bytes:
        """Extract the watermark payload from a watermarked video.

        This is a blind extraction -- the original video is not required.

        Args:
            watermarked_path: Path to the watermarked video file.

        Returns:
            The extracted binary payload.

        Raises:
            FileNotFoundError: If the video file does not exist.
            RuntimeError: If video decoding fails.
        """
        logger.info("Extracting watermark from %s", watermarked_path)

        # 1. Decode the watermarked video into frames.
        frames, _fps, _w, _h = self._read_video_frames(watermarked_path)

        # 2. Process frames through the decoder in batches.
        all_predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(frames), self.batch_size):
                batch_frames = frames[start : start + self.batch_size]
                batch_tensor = self._frames_to_tensor(batch_frames)

                # Decoder outputs per-frame payload probabilities.
                predictions = self.decoder(batch_tensor)  # (B, payload_length)
                all_predictions.append(predictions)

        # 3. Majority vote across all frames for each bit position.
        # Stack all predictions: (total_frames, payload_length).
        stacked = torch.cat(all_predictions, dim=0)
        # Average the probabilities and threshold at 0.5.
        mean_predictions = stacked.mean(dim=0)  # (payload_length,)

        # 4. Convert binary tensor back to bytes.
        extracted = self._tensor_to_bytes(mean_predictions)

        logger.info("Extracted payload: %d bytes", len(extracted))
        return extracted

    def save_weights(self, path: str) -> None:
        """Save encoder and decoder weights to a checkpoint file.

        Args:
            path: Destination file path (typically ``*.pt``).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            path,
        )
        logger.info("Model weights saved to %s", path)
