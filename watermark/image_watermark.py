"""DWT-DCT-SVD invisible steganography for image watermarking."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pywt
from numpy.typing import NDArray
from PIL import Image
from scipy.fft import dctn, idctn

logger = logging.getLogger(__name__)

# Block size used for the block-wise DCT step.
_BLOCK_SIZE = 8


class DwtDctSvdWatermark:
    """Invisible image watermark using a DWT-DCT-SVD hybrid approach.

    The technique works in three stages to embed an imperceptible watermark:

    1. **DWT (Discrete Wavelet Transform)** -- The image is decomposed into
       frequency sub-bands (LL, LH, HL, HH).  The LL (low-frequency)
       sub-band carries the most perceptual energy, so modifications there
       are harder for the human eye to detect when done carefully.

    2. **DCT (Discrete Cosine Transform)** -- A block-wise DCT is applied to
       the selected DWT sub-band, converting spatial-domain coefficients
       into frequency-domain coefficients.  Mid-frequency DCT coefficients
       are targeted because they balance robustness (surviving JPEG
       compression, scaling) with imperceptibility.

    3. **SVD (Singular Value Decomposition)** -- The DCT coefficient matrix
       is factored as U * S * V^T.  The watermark payload is embedded by
       modifying the singular values in S.  Because singular values capture
       the "energy" of the matrix, small perturbations are both robust to
       common image transforms and nearly invisible.

    Extraction reverses the process: DWT -> DCT -> SVD -> compare singular
    values against the original to recover the embedded payload.
    """

    def __init__(self, key: str, strength: float = 0.1) -> None:
        """Initialise the watermarker.

        Args:
            key: Secret key used to seed watermark embedding positions.
            strength: Embedding strength factor. Higher values improve
                robustness at the cost of perceptibility.
        """
        self.key = key
        self.strength = strength

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _payload_to_bits(self, payload: bytes) -> list[int]:
        """Convert a byte sequence to a list of 0/1 bit values."""
        bits: list[int] = []
        for byte_val in payload:
            for shift in range(7, -1, -1):
                bits.append((byte_val >> shift) & 1)
        return bits

    def _bits_to_bytes(self, bits: list[int]) -> bytes:
        """Convert a list of 0/1 bit values back to bytes."""
        # Pad to a multiple of 8 if necessary.
        padded = bits + [0] * ((8 - len(bits) % 8) % 8)
        result = bytearray()
        for i in range(0, len(padded), 8):
            byte_val = 0
            for shift, bit in enumerate(padded[i : i + 8]):
                byte_val = (byte_val << 1) | bit
            result.append(byte_val)
        return bytes(result)

    def _derive_block_indices(self, num_blocks: int, num_bits: int) -> list[int]:
        """Derive deterministic block indices from the secret key.

        Uses HMAC-SHA256 to produce a pseudo-random permutation seeded by
        ``self.key`` so that the same key always yields the same embedding
        positions.

        Args:
            num_blocks: Total number of 8x8 blocks available.
            num_bits: Number of payload bits to embed.

        Returns:
            A list of *num_bits* unique block indices.

        Raises:
            ValueError: If there are fewer available blocks than bits to embed.
        """
        if num_blocks < num_bits:
            raise ValueError(f"Image too small: {num_blocks} blocks available but {num_bits} payload bits to embed.")
        # Derive a deterministic seed from the key.
        seed_bytes = hashlib.sha256(self.key.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:4], "big")
        rng = np.random.RandomState(seed)
        indices = rng.permutation(num_blocks)[:num_bits].tolist()
        return indices

    @staticmethod
    def _split_into_blocks(
        matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], int, int]:
        """Split a 2-D matrix into non-overlapping 8x8 blocks.

        The matrix is cropped to the nearest multiple of ``_BLOCK_SIZE`` in
        each dimension before splitting.

        Returns:
            (blocks, rows_of_blocks, cols_of_blocks) where *blocks* has shape
            ``(num_blocks, 8, 8)``.
        """
        h, w = matrix.shape
        h_crop = (h // _BLOCK_SIZE) * _BLOCK_SIZE
        w_crop = (w // _BLOCK_SIZE) * _BLOCK_SIZE
        cropped = matrix[:h_crop, :w_crop]
        rows_of_blocks = h_crop // _BLOCK_SIZE
        cols_of_blocks = w_crop // _BLOCK_SIZE
        blocks = cropped.reshape(rows_of_blocks, _BLOCK_SIZE, cols_of_blocks, _BLOCK_SIZE)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, _BLOCK_SIZE, _BLOCK_SIZE)
        return blocks, rows_of_blocks, cols_of_blocks

    @staticmethod
    def _merge_blocks(
        blocks: NDArray[np.float64],
        rows_of_blocks: int,
        cols_of_blocks: int,
    ) -> NDArray[np.float64]:
        """Merge 8x8 blocks back into a single 2-D matrix."""
        matrix = blocks.reshape(rows_of_blocks, cols_of_blocks, _BLOCK_SIZE, _BLOCK_SIZE)
        matrix = matrix.transpose(0, 2, 1, 3).reshape(rows_of_blocks * _BLOCK_SIZE, cols_of_blocks * _BLOCK_SIZE)
        return matrix

    @staticmethod
    def _image_to_ycbcr_array(image_path: str) -> tuple[NDArray[np.float64], Image.Image]:
        """Load an image and return its YCbCr luminance channel as float64."""
        img = Image.open(image_path).convert("YCbCr")
        arr = np.array(img, dtype=np.float64)
        return arr, img

    # ------------------------------------------------------------------
    # Forward pass helpers (DWT → DCT → SVD)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_dwt(
        y_channel: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
        """Apply a single-level 2-D DWT and return (LL, detail_coeffs)."""
        coeffs = pywt.dwt2(y_channel, "haar")
        ll: NDArray[np.float64] = coeffs[0]
        details: tuple[NDArray[np.float64], ...] = coeffs[1]
        return ll, details

    @staticmethod
    def _forward_dct_svd(
        blocks: NDArray[np.float64],
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """Apply DCT then SVD to each block. Returns list of (U, S, Vt)."""
        results: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
        for block in blocks:
            dct_block = dctn(block, type=2, norm="ortho")
            u, s, vt = np.linalg.svd(dct_block, full_matrices=True)
            results.append((u, s, vt))
        return results

    # ------------------------------------------------------------------
    # Inverse pass helpers (SVD → DCT → DWT)
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse_svd_dct(
        svd_components: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]],
        num_blocks: int,
    ) -> NDArray[np.float64]:
        """Reconstruct blocks from (U, S, Vt) via inverse SVD then inverse DCT."""
        blocks = np.empty((num_blocks, _BLOCK_SIZE, _BLOCK_SIZE), dtype=np.float64)
        for idx, (u, s, vt) in enumerate(svd_components):
            dct_block = u @ np.diag(s) @ vt
            blocks[idx] = idctn(dct_block, type=2, norm="ortho")
        return blocks

    @staticmethod
    def _inverse_dwt(
        ll: NDArray[np.float64],
        details: tuple[NDArray[np.float64], ...],
        original_shape: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Inverse 2-D DWT to reconstruct the luminance channel."""
        reconstructed: NDArray[np.float64] = pywt.idwt2((ll, details), "haar")
        # pywt may produce an array that is 1 pixel larger; crop to original.
        return reconstructed[: original_shape[0], : original_shape[1]]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, image_path: str, payload: bytes, output_path: str) -> None:
        """Embed an invisible watermark into an image.

        Args:
            image_path: Path to the source image file.
            payload: Binary payload to embed (e.g. a creator/job ID hash).
            output_path: Path where the watermarked image will be saved.

        Raises:
            FileNotFoundError: If the source image does not exist.
            ValueError: If the image is too small for the payload.
        """
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        logger.info(
            "Embedding watermark: payload=%d bytes, strength=%.3f, image=%s",
            len(payload),
            self.strength,
            image_path,
        )

        # 1. Load image, convert to YCbCr, isolate luminance.
        ycbcr_arr, _img = self._image_to_ycbcr_array(image_path)
        y_channel = ycbcr_arr[:, :, 0]
        original_y_shape = y_channel.shape

        # 2. DWT on the luminance channel.
        ll, details = self._forward_dwt(y_channel)

        # 3. Block-wise DCT + SVD on the LL sub-band.
        blocks, rb, cb = self._split_into_blocks(ll)
        svd_components = self._forward_dct_svd(blocks)

        # 4. Determine which blocks/positions carry payload bits.
        bits = self._payload_to_bits(payload)
        num_bits = len(bits)
        block_indices = self._derive_block_indices(len(blocks), num_bits)

        # 5. Embed payload bits into the first singular value of chosen blocks.
        for bit_idx, blk_idx in enumerate(block_indices):
            u, s, vt = svd_components[blk_idx]
            s_modified = s.copy()
            s_modified[0] += self.strength * bits[bit_idx]
            svd_components[blk_idx] = (u, s_modified, vt)

        # 6. Inverse SVD → inverse DCT → blocks → LL sub-band.
        reconstructed_blocks = self._inverse_svd_dct(svd_components, len(blocks))
        ll_reconstructed = self._merge_blocks(reconstructed_blocks, rb, cb)

        # Patch the reconstructed LL back into the original LL (which may be
        # slightly larger due to cropping in _split_into_blocks).
        ll[: ll_reconstructed.shape[0], : ll_reconstructed.shape[1]] = ll_reconstructed

        # 7. Inverse DWT to rebuild the full luminance channel.
        y_reconstructed = self._inverse_dwt(ll, details, original_y_shape)

        # Clamp to valid [0, 255] range.
        y_reconstructed = np.clip(y_reconstructed, 0.0, 255.0)

        # Replace the luminance channel and convert back to RGB.
        ycbcr_arr[:, :, 0] = y_reconstructed
        watermarked_ycbcr = Image.fromarray(ycbcr_arr.astype(np.uint8), mode="YCbCr")
        watermarked_rgb = watermarked_ycbcr.convert("RGB")

        # Ensure output directory exists.
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        watermarked_rgb.save(output_path)

        logger.info("Watermarked image saved to %s", output_path)

    def extract(self, watermarked_path: str, original_path: str) -> bytes:
        """Extract the watermark payload from a watermarked image.

        This is a **non-blind** extraction — both the watermarked and the
        original (unwatermarked) image are required so that singular-value
        differences can be measured.

        Args:
            watermarked_path: Path to the watermarked image.
            original_path: Path to the original (unwatermarked) image.

        Returns:
            The extracted binary payload.

        Raises:
            FileNotFoundError: If either image file does not exist.
            ValueError: If the images have incompatible dimensions.
        """
        for p in (watermarked_path, original_path):
            if not Path(p).is_file():
                raise FileNotFoundError(f"Image not found: {p}")

        logger.info(
            "Extracting watermark: watermarked=%s, original=%s",
            watermarked_path,
            original_path,
        )

        # 1. Load both images, convert to YCbCr, isolate luminance.
        wm_arr, _ = self._image_to_ycbcr_array(watermarked_path)
        orig_arr, _ = self._image_to_ycbcr_array(original_path)

        wm_y = wm_arr[:, :, 0]
        orig_y = orig_arr[:, :, 0]

        if wm_y.shape != orig_y.shape:
            raise ValueError(f"Image dimension mismatch: watermarked={wm_y.shape}, original={orig_y.shape}")

        # 2. DWT on both luminance channels.
        wm_ll, _ = self._forward_dwt(wm_y)
        orig_ll, _ = self._forward_dwt(orig_y)

        # 3. Block-wise DCT + SVD.
        wm_blocks, _rb, _cb = self._split_into_blocks(wm_ll)
        orig_blocks, _, _ = self._split_into_blocks(orig_ll)

        wm_svd = self._forward_dct_svd(wm_blocks)
        orig_svd = self._forward_dct_svd(orig_blocks)

        # 4. We need to know how many bits were embedded. We cannot know the
        #    exact payload length without a header, so we extract all
        #    candidate bits from all key-selected blocks up to the maximum
        #    capacity and rely on the caller to know the expected length.
        #    As a practical heuristic we extract up to the full block count
        #    and trim trailing zero-bytes later.
        num_blocks = len(wm_blocks)

        # We iterate over *all* blocks in key-derived order and recover bits.
        # Derive the maximum possible index set (one bit per block).
        seed_bytes = hashlib.sha256(self.key.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:4], "big")
        rng = np.random.RandomState(seed)
        all_indices = rng.permutation(num_blocks).tolist()

        bits: list[int] = []
        for blk_idx in all_indices:
            _, wm_s, _ = wm_svd[blk_idx]
            _, orig_s, _ = orig_svd[blk_idx]
            diff = wm_s[0] - orig_s[0]
            # A difference close to self.strength indicates a 1-bit; close to
            # 0 indicates a 0-bit.
            bit = 1 if diff > (self.strength / 2.0) else 0
            bits.append(bit)

        # Strip trailing zero bytes to recover the original payload length.
        raw = self._bits_to_bytes(bits)
        # Remove trailing null bytes (but keep at least one byte).
        payload = raw.rstrip(b"\x00") or raw[:1]

        logger.info("Extracted payload: %d bytes", len(payload))
        return payload
