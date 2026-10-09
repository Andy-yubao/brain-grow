"""当前数据协议使用的原始前景仿射渲染与裁切审计原语。"""
import numpy as np
from PIL import Image

OUTPUT_SIZE = 28

def foreground(image):
    image = np.asarray(image)
    if image.shape != (28, 28) or image.dtype != np.uint8:
        raise ValueError("原始MNIST必须为28×28 uint8灰度图")
    yy, xx = np.nonzero(image)
    if not len(xx):
        raise ValueError("原始数字前景为空")
    bbox = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    tile = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return Image.fromarray(np.pad(tile, 2)), bbox


def pose_bounds(bbox, scale, poses):
    """实际前景像素格包围盒的旋转边界，不按整张方形图估算。"""
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    angle = np.deg2rad(poses[..., 2])
    half_x = scale * (np.abs(np.cos(angle)) * w + np.abs(np.sin(angle)) * h) / 2
    half_y = scale * (np.abs(np.sin(angle)) * w + np.abs(np.cos(angle)) * h) / 2
    return np.stack((poses[..., 0] - half_x, poses[..., 1] - half_y,
                     poses[..., 0] + half_x, poses[..., 1] + half_y), axis=-1)


def render_pose(tile, scale, pose, supersampling=4, padding=0):
    """从原始前景直接采样一次；padding供独立大画布裁切审计使用。"""
    x, y, angle = pose
    x, y = x + padding, y + padding
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    a, b = c / (scale * supersampling), s / (scale * supersampling)
    d, e = -s / (scale * supersampling), c / (scale * supersampling)
    coefficients = (a, b, tile.width / 2 - (c * x + s * y) / scale,
                    d, e, tile.height / 2 + (s * x - c * y) / scale)
    size = OUTPUT_SIZE + 2 * padding
    transformed = tile.transform((size * supersampling, size * supersampling), Image.Transform.AFFINE,
                                 coefficients, Image.Resampling.BILINEAR, fillcolor=0)
    return np.asarray(transformed.resize((size, size), Image.Resampling.BOX), np.uint8)
