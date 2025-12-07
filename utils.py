from PIL import Image
import io
import base64


def load_and_resize_image(path, max_dim=512) -> str:
    """
    加载图像 → 压缩尺寸（最长边=max_dim） → 压缩JPEG质量，最后输出base64编码。
    llama.cpp 多模态要求图像token越少越好，否则容易OOM或slot错误。
    """
    img = Image.open(path).convert("RGB")
    # w, h = img.size

    # # 自动缩放：最长边 = max_dim
    # scale = max(w, h) / max_dim
    # if scale > 1.0:
    #     img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)

    # 压缩为JPEG，质量85（显著减少token）
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64
