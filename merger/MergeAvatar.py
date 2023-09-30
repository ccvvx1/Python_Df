# 导入必要的库
import cv2
import numpy as np
from core import imagelib
from facelib import FaceType, LandmarksProcessor
from core.cv2ex import *

# 处理帧信息的函数
def process_frame_info(frame_info, inp_sh):
    # 读取图像
    img_uint8 = cv2_imread(frame_info.filename)
    # 归一化通道
    img_uint8 = imagelib.normalize_channels(img_uint8, 3)
    img = img_uint8.astype(np.float32) / 255.0

    # 获取变换矩阵，用于对齐人脸
    img_mat = LandmarksProcessor.get_transform_mat(frame_info.landmarks_list[0], inp_sh[0], face_type=FaceType.FULL_NO_ALIGN)
    # 应用仿射变换
    img = cv2.warpAffine(img, img_mat, inp_sh[0:2], borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return img

# 人脸融合函数
def MergeFaceAvatar(predictor_func, predictor_input_shape, cfg, prev_temporal_frame_infos, frame_info, next_temporal_frame_infos):
    inp_sh = predictor_input_shape

    prev_imgs = []
    next_imgs = []
    for i in range(cfg.temporal_face_count):
        # 处理前一帧和后一帧的图像信息
        prev_imgs.append(process_frame_info(prev_temporal_frame_infos[i], inp_sh))
        next_imgs.append(process_frame_info(next_temporal_frame_infos[i], inp_sh))
    # 处理当前帧的图像信息
    img = process_frame_info(frame_info, inp_sh)

    # 使用预测器函数进行人脸融合
    prd_f = predictor_func(prev_imgs, img, next_imgs)

    # 如果需要超分辨率处理，可以在此处添加代码
    # if cfg.super_resolution_mode != 0:
    #     prd_f = cfg.superres_func(cfg.super_resolution_mode, prd_f)

    # 如果需要锐化处理，可以在此处添加代码
    if cfg.sharpen_mode != 0 and cfg.sharpen_amount != 0:
        prd_f = cfg.sharpen_func(prd_f, cfg.sharpen_mode, 3, cfg.sharpen_amount)

    # 对输出图像进行剪裁
    out_img = np.clip(prd_f, 0.0, 1.0)

    # 如果需要将源图像添加到输出图像中，可以在此处添加代码
    if cfg.add_source_image:
        out_img = np.concatenate([cv2.resize(img, (prd_f.shape[1], prd_f.shape[0])), out_img], axis=1)

    # 将浮点数图像转换为整数类型（0-255范围）
    return (out_img * 255).astype(np.uint8)
