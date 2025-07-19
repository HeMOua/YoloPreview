from typing import List, Dict

from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_boxes_with_pil(
    image: np.ndarray,
    box_data: List[Dict],
    font_path: str = "fonts/simhei.ttf",
    font_size: int = 16,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    使用 PIL 绘制边框和美化的中文标签，支持圆角背景和透明度。
    """
    image_pil = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", image_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 字体加载
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print(f"无法加载字体：{font_path}")
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    for item in box_data:
        x1, y1, x2, y2 = item["box"]
        label = item["label"]
        color = item["color"]
        color_rgba = color + (int(alpha * 255),)

        # 画边框
        width = 3
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        padding = 4
        text_x = x1
        text_y = y1 - text_h - padding * 2
        if text_y < 0:
            text_y = y1 + width
            text_x += width

        # 圆角背景
        bg_rect = [
            (text_x, text_y),
            (text_x + text_w + padding * 2, text_y + text_h + padding * 2)
        ]
        draw.rectangle(bg_rect, fill=color_rgba)

        # 绘制文字
        draw.text((text_x + padding, text_y + padding), label, font=font, fill=(255, 255, 255, 255))

    return np.array(Image.alpha_composite(image_pil, overlay).convert("RGB"))
