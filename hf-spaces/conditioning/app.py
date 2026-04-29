import gradio as gr
import base64
import io
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
from image_conditioner import ImageConditioner

START_TIME = time.time()
VERSION = "1.0.0"


def _decode_b64_image(b64_string: str) -> np.ndarray:
    """Decode base64 (with or without data URL prefix) to BGR ndarray."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    raw = base64.b64decode(b64_string)
    pil = Image.open(io.BytesIO(raw))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    arr = np.asarray(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _encode_bgr_to_b64_png(bgr_image: np.ndarray, compression: int = 2) -> str:
    """Encode BGR or grayscale ndarray to base64 PNG with data URL prefix."""
    ok, buf = cv2.imencode(".png", bgr_image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return f"data:image/png;base64,{base64.b64encode(buf).decode('ascii')}"


def condition_image(
    image_b64: str,
    strength: int = 10,
    adaptive_block_size: int = 11,
    adaptive_C: int = 2,
    morph_iterations: int = 1,
    target_width: int = 1280,
    target_height: int = 1792,
    png_compression: int = 2,
    return_stages: bool = False,
) -> dict:
    """Condition an image for OCR. Returns dict with conditioned image as base64 PNG.
    If return_stages=True, also returns intermediate stage images (each as base64 JPEG).

    NOTE: ImageConditioner.__init__ calls os.makedirs("results_conditioned") on every
    instantiation — this creates a results_conditioned/ folder in the Space working dir.
    Harmless, but worth knowing if you see that folder appear.
    """
    bgr = _decode_b64_image(image_b64)
    conditioner = ImageConditioner(
        strength=int(strength),
        adaptive_block_size=int(adaptive_block_size),
        adaptive_C=int(adaptive_C),
        morph_iterations=int(morph_iterations),
        target_width=int(target_width),
        target_height=int(target_height),
        png_compression=int(png_compression),
    )

    if return_stages:
        final, stages = conditioner.condition_image_array(bgr, return_stages=True)
        stages_b64 = {}
        for name, stage_img in stages.items():
            ok, buf = cv2.imencode(".jpg", stage_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                stages_b64[name] = (
                    f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"
                )
        return {
            "conditioned_image": _encode_bgr_to_b64_png(final, int(png_compression)),
            "stages": stages_b64,
        }

    final = conditioner.condition_image_array(bgr)
    return {"conditioned_image": _encode_bgr_to_b64_png(final, int(png_compression))}


def health_check() -> dict:
    """Liveness probe."""
    return {
        "status": "ok",
        "version": VERSION,
        "uptime_seconds": int(time.time() - START_TIME),
    }


def _ui_condition(
    image_path,
    strength,
    adaptive_block_size,
    adaptive_C,
    morph_iterations,
    target_width,
    target_height,
    png_compression,
):
    """UI wrapper: reads filepath from gr.Image, calls condition_image, returns PIL for display."""
    if image_path is None:
        return None, "No image provided"
    with open(image_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    b64 = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    result = condition_image(
        b64,
        int(strength),
        int(adaptive_block_size),
        int(adaptive_C),
        int(morph_iterations),
        int(target_width),
        int(target_height),
        int(png_compression),
    )

    img_data = base64.b64decode(result["conditioned_image"].split(",", 1)[1])
    pil = Image.open(io.BytesIO(img_data))
    return pil, "Done"


with gr.Blocks(title="J-Nita Conditioning") as demo:
    gr.Markdown("# J-Nita Image Conditioning Service")
    gr.Markdown("Internal service. Upload an image to test conditioning.")

    with gr.Tabs():
        with gr.Tab("Condition"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="filepath", label="Input Image")
                    strength_sl = gr.Slider(
                        0, 100, value=10, step=1,
                        label="Strength  (0 = original, 100 = fully processed)",
                    )
                    with gr.Accordion("Advanced Parameters", open=False):
                        block_size_sl = gr.Slider(
                            3, 51, value=11, step=2, label="Adaptive Block Size (odd)"
                        )
                        adaptive_c_num = gr.Number(value=2, label="Adaptive C")
                        morph_iter_sl = gr.Slider(
                            0, 5, value=1, step=1, label="Morph Iterations"
                        )
                        target_w_num = gr.Number(value=1280, label="Target Width (px)")
                        target_h_num = gr.Number(value=1792, label="Target Height (px)")
                        compression_sl = gr.Slider(
                            0, 9, value=2, step=1, label="PNG Compression"
                        )
                    submit_btn = gr.Button("Condition Image", variant="primary")
                with gr.Column():
                    output_img = gr.Image(label="Conditioned Output")
                    status_out = gr.Textbox(label="Status", interactive=False)

            submit_btn.click(
                fn=_ui_condition,
                inputs=[
                    img_input, strength_sl, block_size_sl, adaptive_c_num,
                    morph_iter_sl, target_w_num, target_h_num, compression_sl,
                ],
                outputs=[output_img, status_out],
                api_name=False,
            )

        with gr.Tab("Health"):
            gr.Markdown("Liveness check — returns service version and uptime.")
            health_btn = gr.Button("Check Health")
            health_out = gr.JSON(label="Health")
            health_btn.click(
                fn=health_check,
                inputs=[],
                outputs=[health_out],
                api_name="health_check",
            )

    # Hidden components that expose condition_image as a named API endpoint.
    # Gradio registers event handlers by api_name regardless of component visibility,
    # so programmatic callers can POST to /api/condition_image with a base64 string.
    with gr.Row(visible=False):
        _b64_in = gr.Textbox()
        _str_in = gr.Number(value=10)
        _blk_in = gr.Number(value=11)
        _c_in = gr.Number(value=2)
        _mor_in = gr.Number(value=1)
        _tw_in = gr.Number(value=1280)
        _th_in = gr.Number(value=1792)
        _comp_in = gr.Number(value=2)
        _stg_in = gr.Checkbox(value=False)
        _json_out = gr.JSON()
        _api_btn = gr.Button()
        _api_btn.click(
            fn=condition_image,
            inputs=[_b64_in, _str_in, _blk_in, _c_in, _mor_in, _tw_in, _th_in, _comp_in, _stg_in],
            outputs=_json_out,
            api_name="condition_image",
        )


if __name__ == "__main__":
    demo.launch()
