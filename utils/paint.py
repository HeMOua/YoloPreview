import os
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



def draw_segmentation_with_pil(image_rgb, segmentation_data):
    """
    使用PIL绘制实例分割掩码

    Args:
        image_rgb: RGB格式的numpy图像数组
        segmentation_data: 分割数据列表，每项包含:
            - segmentation: 多边形点列表 [(x1, y1), (x2, y2), ...]
            - label: 类别标签
            - confidence: 置信度
            - color: RGB颜色元组
            - alpha: 掩码透明度 (0.0-1.0)

    Returns:
        绘制后的RGB图像数组
    """
    # 转换为PIL Image
    pil_image = Image.fromarray(image_rgb)

    # 创建一个透明层用于绘制掩码
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # 加载字体
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "C:\\Windows\\Fonts\\msyh.ttc",
            "C:\\Windows\\Fonts\\simsun.ttc"
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 16)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 转换为RGBA以支持alpha混合
    if pil_image.mode != 'RGBA':
        pil_image = pil_image.convert('RGBA')

    for item in segmentation_data:
        segmentation = item['segmentation']
        label = item['label']
        confidence = item.get('confidence', 1.0)
        color = item['color']
        alpha = item.get('alpha', 0.4)

        if not segmentation or len(segmentation) < 3:
            continue

        # 转换点格式为PIL需要的格式
        points = []
        for point in segmentation:
            points.append((float(point[0]), float(point[1])))

        # 计算alpha通道值
        alpha_value = int(alpha * 255)
        fill_color = (*color, alpha_value)
        outline_color = (*color, 255)

        # 绘制填充的多边形（半透明）
        overlay_draw.polygon(points, fill=fill_color, outline=outline_color, width=2)

        # 计算多边形中心点用于放置标签
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        # 准备标签文本
        if confidence < 1.0:
            text = f"{label} {confidence:.2f}"
        else:
            text = label

        # 计算文本框大小
        bbox = overlay_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 绘制标签背景（不透明）
        label_x1 = center_x - text_width / 2 - 2
        label_y1 = center_y - text_height / 2 - 2
        label_x2 = center_x + text_width / 2 + 2
        label_y2 = center_y + text_height / 2 + 2

        overlay_draw.rectangle(
            [(label_x1, label_y1), (label_x2, label_y2)],
            fill=(*color, 200)
        )

        # 绘制标签文本
        overlay_draw.text(
            (label_x1 + 2, label_y1 + 2),
            text,
            fill=(255, 255, 255, 255),
            font=font
        )

    # 将overlay层合并到原图
    result = Image.alpha_composite(pil_image, overlay)

    # 转换回RGB
    result = result.convert('RGB')

    # 转换回numpy数组
    return np.array(result)
