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


def crop_and_reassemble(path, keep=("tl","tr")) -> str:
    """
    将图像切成九宫格（3x3），保留指定区域（keep），再拼接成新图。
    keep 可选元素：
        'top', 'center', 'bottom'
        以及细分:
        'tl','tc','tr','cl','cc','cr','bl','bc','br'

    返回：base64 JPEG 字符串
    """

    img = Image.open(path).convert("RGB")
    w, h = img.size

    # 分成3x3的格子
    w3 = w // 3
    h3 = h // 3

    # 每个格子的坐标
    boxes = {
        "tl": (0,      0,      w3,      h3),      # top-left
        "tc": (w3,     0,      2*w3,    h3),      # top-center
        "tr": (2*w3,   0,      w,       h3),      # top-right

        "cl": (0,      h3,     w3,      2*h3),    # center-left
        "cc": (w3,     h3,     2*w3,    2*h3),    # center-center (最重要)
        "cr": (2*w3,   h3,     w,       2*h3),    # center-right

        "bl": (0,      2*h3,   w3,      h),       # bottom-left
        "bc": (w3,     2*h3,   2*w3,    h),       # bottom-center
        "br": (2*w3,   2*h3,   w,       h),       # bottom-right
    }

    # keep 支持简写：'top','center','bottom'
    alias = {
        "top":    ["tl", "tc", "tr"],
        "center": ["cl", "cc", "cr"],
        "bottom": ["bl", "bc", "br"],
    }

    expanded_keep = []
    for k in keep:
        expanded_keep.extend(alias.get(k, [k]))

    crops = [img.crop(boxes[k]) for k in expanded_keep]

    # 将裁剪的块拼成一个长条（简单做法）
    new_w = sum(c.width for c in crops)
    new_h = max(c.height for c in crops)

    new_img = Image.new("RGB", (new_w, new_h))
    x_offset = 0
    for c in crops:
        new_img.paste(c, (x_offset, 0))
        x_offset += c.width

    print("Original:", img.size, " → Cropped & merged:", new_img.size)

    # 转成 base64
    buf = io.BytesIO()
    new_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return b64
