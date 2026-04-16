"""
MCP Tool Definitions — Image Processing Capabilities.
Each tool is a discrete, discoverable capability with a schema.
Tools are registered at startup; agents discover them dynamically.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2

from protocols import MCPRegistry, MCPTool


def _pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ─── Tool Handlers ────────────────────────────────────────────────────────────

def handle_denoise(image_b64: str, strength: float = 0.5) -> dict:
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img)
    # OpenCV expects BGR for fastNlMeansDenoisingColored
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h = int(3 + strength * 7)  # 3–10
    h = h if h % 2 == 1 else h + 1
    denoised_bgr = cv2.fastNlMeansDenoisingColored(arr_bgr, None, h, h, 7, 21)
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(denoised_rgb)
    return {"image_b64": _pil_to_b64(result), "info": f"Applied NL-means denoising (h={h})"}


def handle_enhance_contrast(image_b64: str, factor: float = 1.5) -> dict:
    img = _b64_to_pil(image_b64)
    enhanced = ImageEnhance.Contrast(img).enhance(factor)
    return {"image_b64": _pil_to_b64(enhanced), "info": f"Contrast enhanced by {factor}x"}


def handle_sharpen(image_b64: str, radius: float = 2.0, percent: int = 150) -> dict:
    img = _b64_to_pil(image_b64)
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
    return {"image_b64": _pil_to_b64(sharpened), "info": f"Unsharp mask (r={radius}, p={percent}%)"}


def handle_resize(image_b64: str, width: int = 512, height: int = 512, keep_aspect: bool = True) -> dict:
    img = _b64_to_pil(image_b64)
    if keep_aspect:
        img.thumbnail((width, height), Image.Resampling.LANCZOS)
        result = img
    else:
        result = img.resize((width, height), Image.Resampling.LANCZOS)
    return {"image_b64": _pil_to_b64(result), "info": f"Resized to {result.size[0]}x{result.size[1]}"}


def handle_grayscale(image_b64: str) -> dict:
    img = _b64_to_pil(image_b64).convert("L").convert("RGB")
    return {"image_b64": _pil_to_b64(img), "info": "Converted to grayscale"}


def handle_edge_detection(image_b64: str, low_threshold: int = 50, high_threshold: int = 150) -> dict:
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(edges_rgb)
    return {"image_b64": _pil_to_b64(result), "info": f"Canny edge detection ({low_threshold}-{high_threshold})"}


def handle_histogram_equalization(image_b64: str) -> dict:
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img)
    img_yuv = cv2.cvtColor(arr, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    result = Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))
    return {"image_b64": _pil_to_b64(result), "info": "Histogram equalization applied"}


def handle_color_correction(image_b64: str, temperature: float = 0.0, tint: float = 0.0) -> dict:
    """temperature: -1 (cool/blue) to +1 (warm/orange). tint: -1 (green) to +1 (magenta)."""
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    # Warm/cool shift on R and B channels
    arr[:, :, 0] = np.clip(arr[:, :, 0] + temperature * 30, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] - temperature * 30, 0, 255)
    # Tint on G channel
    arr[:, :, 1] = np.clip(arr[:, :, 1] + tint * 20, 0, 255)
    result = Image.fromarray(arr.astype(np.uint8))
    return {"image_b64": _pil_to_b64(result), "info": f"Color correction (temp={temperature:+.1f}, tint={tint:+.1f})"}


def handle_color_grading(
    image_b64: str,
    preset: str = "cinematic",
    intensity: float = 0.7,
    warmth: float = 0.0,
    saturation: float = 1.0,
    contrast: float = 1.0,
    lift: float = 0.0,
    gamma: float = 1.0,
) -> dict:
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    preset_key = (preset or "cinematic").lower()
    intensity = float(np.clip(intensity, 0.0, 1.0))

    looks = {
        "cinematic": {"contrast": 1.15, "saturation": 0.88, "warmth": 0.08, "lift": -0.03, "gamma": 0.96},
        "vivid": {"contrast": 1.10, "saturation": 1.22, "warmth": 0.02, "lift": 0.00, "gamma": 1.00},
        "warm_film": {"contrast": 1.05, "saturation": 0.96, "warmth": 0.14, "lift": 0.03, "gamma": 0.98},
        "soft_film": {"contrast": 0.92, "saturation": 0.90, "warmth": 0.03, "lift": 0.06, "gamma": 1.05},
        "noir": {"contrast": 1.25, "saturation": 0.0, "warmth": 0.0, "lift": -0.08, "gamma": 0.92},
    }
    look = looks.get(preset_key, looks["cinematic"])

    grade_contrast = 1.0 + (look["contrast"] - 1.0) * intensity + (contrast - 1.0) * intensity
    grade_saturation = max(0.0, 1.0 + (look["saturation"] - 1.0) * intensity + (saturation - 1.0) * intensity)
    grade_warmth = look["warmth"] * intensity + warmth * intensity
    grade_lift = look["lift"] * intensity + lift * intensity
    grade_gamma = max(0.2, look["gamma"] * (1.0 - intensity) + gamma * intensity)

    # Contrast around mid-gray.
    arr = (arr - 0.5) * grade_contrast + 0.5

    # Basic split-toning: warm highlights, cool shadows.
    highlights = np.clip((arr - 0.5) * 1.4 + 0.5, 0.0, 1.0)
    shadows = np.clip((0.5 - arr) * 1.4 + 0.5, 0.0, 1.0)
    arr[:, :, 0] += grade_warmth * 0.10 * highlights[:, :, 0]
    arr[:, :, 2] -= grade_warmth * 0.10 * highlights[:, :, 2]
    arr[:, :, 0] -= grade_warmth * 0.06 * shadows[:, :, 0]
    arr[:, :, 2] += grade_warmth * 0.06 * shadows[:, :, 2]

    # Lift/lower shadows and highlights a bit for a graded feel.
    arr = np.clip(arr + grade_lift, 0.0, 1.0)

    # Gamma adjustment.
    arr = np.clip(arr, 0.0, 1.0) ** (1.0 / grade_gamma)

    graded = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    graded = ImageEnhance.Color(graded).enhance(grade_saturation)

    return {
        "image_b64": _pil_to_b64(graded),
        "info": (
            f"Color graded with {preset_key} look "
            f"(intensity={intensity:.2f}, sat={grade_saturation:.2f}, contrast={grade_contrast:.2f})"
        ),
    }


def handle_brightness(image_b64: str, factor: float = 1.2) -> dict:
    img = _b64_to_pil(image_b64)
    result = ImageEnhance.Brightness(img).enhance(factor)
    return {"image_b64": _pil_to_b64(result), "info": f"Brightness adjusted by {factor}x"}


def handle_saturation(image_b64: str, factor: float = 1.3) -> dict:
    img = _b64_to_pil(image_b64)
    result = ImageEnhance.Color(img).enhance(factor)
    return {"image_b64": _pil_to_b64(result), "info": f"Saturation adjusted by {factor}x"}


def handle_rotate(image_b64: str, angle: float = 0.0, expand: bool = True) -> dict:
    img = _b64_to_pil(image_b64)
    result = img.rotate(-angle, expand=expand, resample=Image.Resampling.BICUBIC)
    return {"image_b64": _pil_to_b64(result), "info": f"Rotated by {angle}°"}


def handle_crop_center(image_b64: str, crop_pct: float = 0.8) -> dict:
    img = _b64_to_pil(image_b64)
    w, h = img.size
    new_w, new_h = int(w * crop_pct), int(h * crop_pct)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    result = img.crop((left, top, left + new_w, top + new_h))
    return {"image_b64": _pil_to_b64(result), "info": f"Center-cropped to {crop_pct*100:.0f}% of original"}


def handle_blur(image_b64: str, radius: float = 2.0) -> dict:
    img = _b64_to_pil(image_b64)
    result = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return {"image_b64": _pil_to_b64(result), "info": f"Gaussian blur (radius={radius})"}


def handle_vignette(image_b64: str, strength: float = 0.5) -> dict:
    img = _b64_to_pil(image_b64).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    dist = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1 - dist * strength, 0, 1)
    arr[:, :, 0] *= vignette
    arr[:, :, 1] *= vignette
    arr[:, :, 2] *= vignette
    result = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return {"image_b64": _pil_to_b64(result), "info": f"Vignette applied (strength={strength})"}


# ─── Registry Builder ─────────────────────────────────────────────────────────

def build_mcp_registry() -> MCPRegistry:
    registry = MCPRegistry()
    
    tools = [
        MCPTool("denoise", "Remove noise and grain from images using Non-Local Means algorithm",
                {"strength": {"type": "number", "default": 0.5, "description": "0=light, 1=heavy"}},
                handle_denoise, tags=["quality", "cleanup"]),
        
        MCPTool("enhance_contrast", "Increase or decrease image contrast",
                {"factor": {"type": "number", "default": 1.5}},
                handle_enhance_contrast, tags=["quality", "enhancement"]),
        
        MCPTool("sharpen", "Sharpen image details using unsharp mask",
                {"radius": {"type": "number", "default": 2.0}, "percent": {"type": "integer", "default": 150}},
                handle_sharpen, tags=["quality", "detail"]),
        
        MCPTool("resize", "Resize image to target dimensions",
                {"width": {"type": "integer", "default": 512}, "height": {"type": "integer", "default": 512},
                 "keep_aspect": {"type": "boolean", "default": True}},
                handle_resize, tags=["transform", "size"]),
        
        MCPTool("grayscale", "Convert image to grayscale",
                {}, handle_grayscale, tags=["transform", "color"]),
        
        MCPTool("edge_detection", "Detect edges using Canny algorithm",
                {"low_threshold": {"type": "integer", "default": 50},
                 "high_threshold": {"type": "integer", "default": 150}},
                handle_edge_detection, tags=["analysis", "feature"]),
        
        MCPTool("histogram_equalization", "Equalize histogram for better contrast distribution",
                {}, handle_histogram_equalization, tags=["quality", "enhancement"]),
        
        MCPTool("color_correction", "Adjust white balance, temperature and tint",
                {"temperature": {"type": "number", "default": 0.0, "description": "-1=cool +1=warm"},
                 "tint": {"type": "number", "default": 0.0}},
                handle_color_correction, tags=["color", "correction"]),

        MCPTool("color_grading", "Apply a cinematic or stylized color grade",
            {"preset": {"type": "string", "default": "cinematic", "description": "cinematic, vivid, warm_film, soft_film, noir"},
             "intensity": {"type": "number", "default": 0.7},
             "warmth": {"type": "number", "default": 0.0},
             "saturation": {"type": "number", "default": 1.0},
             "contrast": {"type": "number", "default": 1.0},
             "lift": {"type": "number", "default": 0.0},
             "gamma": {"type": "number", "default": 1.0}},
            handle_color_grading, tags=["color", "grading", "creative"]),
        
        MCPTool("brightness", "Adjust image brightness",
                {"factor": {"type": "number", "default": 1.2}},
                handle_brightness, tags=["quality", "enhancement"]),
        
        MCPTool("saturation", "Adjust color saturation",
                {"factor": {"type": "number", "default": 1.3}},
                handle_saturation, tags=["color", "enhancement"]),
        
        MCPTool("rotate", "Rotate image by specified angle",
                {"angle": {"type": "number", "default": 0.0}, "expand": {"type": "boolean", "default": True}},
                handle_rotate, tags=["transform", "geometry"]),
        
        MCPTool("crop_center", "Crop image to center region",
                {"crop_pct": {"type": "number", "default": 0.8}},
                handle_crop_center, tags=["transform", "composition"]),
        
        MCPTool("blur", "Apply Gaussian blur",
                {"radius": {"type": "number", "default": 2.0}},
                handle_blur, tags=["effect", "smoothing"]),
        
        MCPTool("vignette", "Add dark vignette around image edges",
                {"strength": {"type": "number", "default": 0.5}},
                handle_vignette, tags=["effect", "artistic"]),
    ]
    
    for tool in tools:
        registry.register(tool)
    
    return registry