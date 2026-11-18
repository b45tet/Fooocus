import numpy as np


def editor_to_inpaint(value):
    """Translate Gradio's RGBA editor layers into Fooocus's RGB image and mask."""
    if value is None or value["background"] is None:
        return None

    image = value["background"][:, :, :3].copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for layer in value["layers"]:
        np.maximum(mask, layer[:, :, 3], out=mask)
    return {"image": image, "mask": np.repeat(mask[:, :, None], 3, axis=2)}
