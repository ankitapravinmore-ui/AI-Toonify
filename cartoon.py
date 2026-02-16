import cv2
import numpy as np
import onnxruntime as ort
import os

# ================== LOAD MODEL ==================
_MODEL_PATH = os.path.join("models", "cartoonizer.onnx")
_SESSION = None
_IN_NAME = None
_OUT_NAME = None


def _get_session():
    global _SESSION, _IN_NAME, _OUT_NAME
    if _SESSION is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {_MODEL_PATH}")

        _SESSION = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
        _IN_NAME = _SESSION.get_inputs()[0].name
        _OUT_NAME = _SESSION.get_outputs()[0].name

    return _SESSION, _IN_NAME, _OUT_NAME


# ================== PREPROCESS ==================
def _preprocess(img_rgb):
    # model expects BGR, NHWC, [-1,1], multiple of 8
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    h, w = (h // 8) * 8, (w // 8) * 8
    bgr = bgr[:h, :w, :]

    x = bgr.astype(np.float32) / 127.5 - 1.0
    x = np.expand_dims(x, axis=0)
    return x


# ================== POSTPROCESS ==================
def _postprocess(out):
    y = np.squeeze(out)
    y = (y + 1.0) * 127.5
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y  # BGR


# ================== CARTOON ==================
def cartoon(img_rgb):
    sess, in_name, out_name = _get_session()

    inp = _preprocess(img_rgb)
    out = sess.run(None, {in_name: inp})[0]
    out_bgr = _postprocess(out)

    return out_bgr


# ================== SKETCH ==================
def sketch(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    inv_blur = 255 - blur

    sketch_img = cv2.divide(gray, inv_blur, scale=256.0)
    return cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR)


# ================== PENCIL ==================
def pencil(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
