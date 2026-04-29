---
title: J-Nita Conditioning
emoji: 🖼️
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# J-Nita Image Conditioning Service

Internal preprocessing service for the J-Nita handwriting OCR pipeline.

Accepts a raw image (photo, scan, or document) and returns a conditioned version
optimised for downstream OCR: page detection, perspective correction, adaptive
thresholding, and morphological noise reduction.

## Endpoints

| Endpoint | Description |
|---|---|
| `/api/condition_image` | Condition an image supplied as base64. Returns conditioned image as base64 PNG. |
| `/api/health_check` | Liveness probe — returns `{status, version, uptime_seconds}`. |

## Parameters (`condition_image`)

| Name | Type | Default | Notes |
|---|---|---|---|
| `image_b64` | string | — | Base64 image with optional `data:…;base64,` prefix |
| `strength` | int | 10 | 0 = original resized, 100 = fully processed |
| `adaptive_block_size` | int | 11 | Must be odd |
| `adaptive_C` | int | 2 | Adaptive threshold constant |
| `morph_iterations` | int | 1 | Morphological operation iterations |
| `target_width` | int | 1280 | Max output width in pixels |
| `target_height` | int | 1792 | Max output height in pixels |
| `png_compression` | int | 2 | PNG compression level (0–9) |
| `return_stages` | bool | false | Also return intermediate stage images as base64 JPEG |

## Hardware

CPU Basic (free tier). Hardware and EU region are configured in the HF Space settings UI.
