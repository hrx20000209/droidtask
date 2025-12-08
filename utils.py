from PIL import Image
import io
import base64


def load_and_resize_image(path, max_dim=512) -> str:
    """
    加载图像 → 长宽缩小一半 → JPEG 压缩 → base64。
    llama.cpp 对图片 token 数敏感，缩小一半可以显著降低图像 token。
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # ↓↓↓ 新增：长宽都缩小一半 ↓↓↓
    new_w = max(1, w // 4)
    new_h = max(1, h // 4)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    print(f"Image Size: {new_w}x{new_h}")
    # ↑↑↑ 新增部分 ↑↑↑

    # JPEG 压缩（85 是折中选择）
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64

