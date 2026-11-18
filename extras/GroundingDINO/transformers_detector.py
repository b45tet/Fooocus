from functools import lru_cache

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from ldm_patched.modules import model_management
from ldm_patched.modules.model_patcher import ModelPatcher
from modules.config import path_inpaint


MODEL_ID = "IDEA-Research/grounding-dino-tiny"


@lru_cache(maxsize=1)
def load_detector():
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=path_inpaint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, cache_dir=path_inpaint, use_safetensors=True,
    ).eval()
    model.config.disable_custom_kernels = True
    return processor, ModelPatcher(
        model, load_device=model_management.text_encoder_device(),
        offload_device=model_management.text_encoder_offload_device(),
    )


@torch.inference_mode()
def detect_boxes(image, caption, box_threshold, text_threshold):
    processor, patcher = load_detector()
    model_management.load_model_gpu(patcher)
    text = caption.strip().lower().rstrip(".") + "."
    inputs = processor(images=image, text=text, return_tensors="pt").to(patcher.load_device)
    results = processor.post_process_grounded_object_detection(
        patcher.model(**inputs), inputs.input_ids,
        threshold=box_threshold, text_threshold=text_threshold, target_sizes=[image.shape[:2]],
    )
    return results[0]["boxes"].cpu()
