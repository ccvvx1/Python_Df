# 导入所需库
import sys
import traceback
import cv2
import numpy as np
from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor

# 判断操作系统是否为Windows
is_windows = sys.platform[0:3] == 'win'

# 定义输入的XSeg尺寸
xseg_input_size = 256

# 合成遮挡的人脸
def MergeMaskedFace(predictor_func, predictor_input_shape,
                     face_enhancer_func, 
                     xseg_256_extract_func, 
                     cfg, frame_info, img_bgr_uint8, img_bgr, img_face_landmarks):
    # 获取图像的尺寸
    img_size = img_bgr.shape[1], img_bgr.shape[0]
    # 根据面部特征点生成面部蒙版
    img_face_mask_a = LandmarksProcessor.get_image_hull_mask(img_bgr.shape, img_face_landmarks)

    # 定义输入尺寸、蒙版子采样尺寸和输出尺寸
    input_size = predictor_input_shape[0]
    mask_subres_size = input_size * 4
    output_size = input_size
    if cfg.super_resolution_power != 0:
        output_size *= 4

    # 获取面部变换矩阵，用于对齐面部
    face_mat = LandmarksProcessor.get_transform_mat(img_face_landmarks, output_size, face_type=cfg.face_type)
    face_output_mat = LandmarksProcessor.get_transform_mat(img_face_landmarks, output_size, face_type=cfg.face_type, scale=1.0 + 0.01 * cfg.output_face_scale)

    if mask_subres_size == output_size:
        face_mask_output_mat = face_output_mat
    else:
        face_mask_output_mat = LandmarksProcessor.get_transform_mat(img_face_landmarks, mask_subres_size, face_type=cfg.face_type, scale=1.0 + 0.01 * cfg.output_face_scale)

    # 对面部图像进行仿射变换
    dst_face_bgr = cv2.warpAffine(img_bgr, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
    dst_face_bgr = np.clip(dst_face_bgr, 0, 1)

    # 对面部蒙版进行仿射变换
    dst_face_mask_a_0 = cv2.warpAffine(img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    # 对预测器输入的面部图像进行缩放
    predictor_input_bgr = cv2.resize(dst_face_bgr, (input_size, input_size))

    # 使用预测器函数预测人脸
    predicted = predictor_func(predictor_input_bgr)
    prd_face_bgr = np.clip(predicted[0], 0, 1.0)
    prd_face_mask_a_0 = np.clip(predicted[1], 0, 1.0)
    prd_face_dst_mask_a_0 = np.clip(predicted[2], 0, 1.0)

    # 如果需要超分辨率处理
    if cfg.super_resolution_power != 0:
        prd_face_bgr_enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
        mod = cfg.super_resolution_power / 100.0
        prd_face_bgr = cv2.resize(prd_face_bgr, (output_size, output_size)) * (1.0 - mod) + prd_face_bgr_enhanced * mod
        prd_face_bgr = np.clip(prd_face_bgr, 0, 1)

    # 如果需要超分辨率处理，对相关蒙版也进行缩放
    if cfg.super_resolution_power != 0:
        prd_face_mask_a_0 = cv2.resize(prd_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        prd_face_dst_mask_a_0 = cv2.resize(prd_face_dst_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    # 根据配置中的 mask_mode 不同进行不同的蒙版处理
    if cfg.mask_mode == 0:  # full
        wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
    elif cfg.mask_mode == 1:  # dst
        wrk_face_mask_a_0 = cv2.resize(dst_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    elif cfg.mask_mode == 2:  # learned-prd
        wrk_face_mask_a_0 = prd_face_mask_a_0
    elif cfg.mask_mode == 3:  # learned-dst
        wrk_face_mask_a_0 = prd_face_dst_mask_a_0
    elif cfg.mask_mode == 4:  # learned-prd*learned-dst
        wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0
    elif cfg.mask_mode == 5:  # learned-prd+learned-dst
        wrk_face_mask_a_0 = np.clip(prd_face_mask_a_0 + prd_face_dst_mask_a_0, 0, 1)
    elif cfg.mask_mode >= 6 and cfg.mask_mode <= 9:  # XSeg 模式
        if cfg.mask_mode == 6 or cfg.mask_mode == 8 or cfg.mask_mode == 9:
            # 获取 XSeg-prd
            prd_face_xseg_bgr = cv2.resize(prd_face_bgr, (xseg_input_size,) * 2, interpolation=cv2.INTER_CUBIC)
            prd_face_xseg_mask = xseg_256_extract_func(prd_face_xseg_bgr)
            X_prd_face_mask_a_0 = cv2.resize(prd_face_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        if cfg.mask_mode >= 7 and cfg.mask_mode <= 9:
            # 获取 XSeg-dst
            xseg_mat = LandmarksProcessor.get_transform_mat(img_face_landmarks, xseg_input_size, face_type=cfg.face_type)
            dst_face_xseg_bgr = cv2.warpAffine(img_bgr, xseg_mat, (xseg_input_size,) * 2, flags=cv2.INTER_CUBIC)
            dst_face_xseg_mask = xseg_256_extract_func(dst_face_xseg_bgr)
            X_dst_face_mask_a_0 = cv2.resize(dst_face_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        if cfg.mask_mode == 6:  # 'XSeg-prd'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0
        elif cfg.mask_mode == 7:  # 'XSeg-dst'
            wrk_face_mask_a_0 = X_dst_face_mask_a_0
        elif cfg.mask_mode == 8:  # 'XSeg-prd*XSeg-dst'
            wrk_face_mask_a_0 = X_prd_face_mask_a_0 * X_dst_face_mask_a_0
        elif cfg.mask_mode == 9:  # 'learned-prd*learned-dst*XSeg-prd*XSeg-dst'
            wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0 * X_prd_face_mask_a_0 * X_dst_face_mask_a_0

    # 去除蒙版中的噪音
    wrk_face_mask_a_0[wrk_face_mask_a_0 < (1.0/255.0)] = 0.0

    # 如果蒙版尺寸不等于 mask_subres_size，则进行缩放
    if wrk_face_mask_a_0.shape[0] != mask_subres_size:
        wrk_face_mask_a_0 = cv2.resize(wrk_face_mask_a_0, (mask_subres_size, mask_subres_size), interpolation=cv2.INTER_CUBIC)

    # 根据配置中的模式进行蒙版处理
    if 'raw' not in cfg.mode:
        # 添加零填充
        wrk_face_mask_a_0 = np.pad(wrk_face_mask_a_0, input_size)

        ero = cfg.erode_mask_modifier
        blur = cfg.blur_mask_modifier

        if ero > 0:
            wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero)), iterations=1)
        elif ero < 0:
            wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-ero, -ero)), iterations=1)

        # 在实际预测区域内截取蒙版
        # 在边界处添加半个模糊大小，以准确地淡化到零
        clip_size = input_size + blur // 2

        wrk_face_mask_a_0[:clip_size, :] = 0
        wrk_face_mask_a_0[-clip_size:, :] = 0
        wrk_face_mask_a_0[:, :clip_size] = 0
        wrk_face_mask_a_0[:, -clip_size:] = 0

        if blur > 0:
            blur = blur + (1 - blur % 2)
            wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur), 0)

        wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size, input_size:-input_size]

        wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

    # 将蒙版根据 face_mask_output_mat 映射回原始图像尺寸
    img_face_mask_a = cv2.warpAffine(wrk_face_mask_a_0, face_mask_output_mat, img_size, np.zeros(img_bgr.shape[0:2], dtype=np.float32), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)[..., None]
    img_face_mask_a = np.clip(img_face_mask_a, 0.0, 1.0)
    img_face_mask_a[img_face_mask_a < (1.0/255.0)] = 0.0  # 去除噪音

    # 如果蒙版尺寸不等于输出尺寸，则进行缩放
    if wrk_face_mask_a_0.shape[0] != output_size:
        wrk_face_mask_a_0 = cv2.resize(wrk_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

    # 将蒙版赋值给输出的 img_face_mask_a
    wrk_face_mask_a = wrk_face_mask_a_0[..., None]

    out_img = None
    out_merging_mask_a = None

    # 根据配置的不同模式进行输出
    if cfg.mode == 'original':
        return img_bgr, img_face_mask_a


    # 如果配置模式中包含 'raw'
    elif 'raw' in cfg.mode:
        if cfg.mode == 'raw-rgb':
            # 使用逆映射将合成的预测人脸图像放回原始图像
            out_img_face = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
            out_img_face_mask = cv2.warpAffine(np.ones_like(prd_face_bgr), face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
            # 使用蒙版将合成的人脸与原始图像融合
            out_img = img_bgr * (1 - out_img_face_mask) + out_img_face * out_img_face_mask
            out_merging_mask_a = img_face_mask_a
        elif cfg.mode == 'raw-predict':
            # 直接使用预测的人脸作为输出图像
            out_img = prd_face_bgr
            out_merging_mask_a = wrk_face_mask_a
        else:
            raise ValueError(f"未定义的 raw 模式 {cfg.mode}")

        # 确保输出图像值在 [0, 1] 范围内
        out_img = np.clip(out_img, 0.0, 1.0)
    else:
        # 如果蒙版满足最小尺寸要求，执行以下处理
        maxregion = np.argwhere(img_face_mask_a >= 0.1)
        if maxregion.size != 0:
            miny, minx = maxregion.min(axis=0)[:2]
            maxy, maxx = maxregion.max(axis=0)[:2]
            lenx = maxx - minx
            leny = maxy - miny
            if min(lenx, leny) >= 4:
                wrk_face_mask_area_a = wrk_face_mask_a.copy()
                wrk_face_mask_area_a[wrk_face_mask_area_a > 0] = 1.0

                # 根据颜色转移模式进行颜色转移
                if 'seamless' not in cfg.mode and cfg.color_transfer_mode != 0:
                    if cfg.color_transfer_mode == 1:  # rct
                        prd_face_bgr = imagelib.reinhard_color_transfer(prd_face_bgr, dst_face_bgr, target_mask=wrk_face_mask_area_a, source_mask=wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 2:  # lct
                        prd_face_bgr = imagelib.linear_color_transfer(prd_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 3:  # mkl
                        prd_face_bgr = imagelib.color_transfer_mkl(prd_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 4:  # mkl-m
                        prd_face_bgr = imagelib.color_transfer_mkl(prd_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 5:  # idt
                        prd_face_bgr = imagelib.color_transfer_idt(prd_face_bgr, dst_face_bgr)
                    elif cfg.color_transfer_mode == 6:  # idt-m
                        prd_face_bgr = imagelib.color_transfer_idt(prd_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)
                    elif cfg.color_transfer_mode == 7:  # sot-m
                        prd_face_bgr = imagelib.color_transfer_sot(prd_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a, steps=10, batch_size=30)
                        prd_face_bgr = np.clip(prd_face_bgr, 0.0, 1.0)
                    elif cfg.color_transfer_mode == 8:  # mix-m
                        prd_face_bgr = imagelib.color_transfer_mix(prd_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)

                # 如果模式是 'hist-match'，进行直方图匹配
                if cfg.mode == 'hist-match':
                    hist_mask_a = np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)

                    if cfg.masked_hist_match:
                        hist_mask_a *= wrk_face_mask_area_a

                    white = (1.0 - hist_mask_a) * np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)

                    hist_match_1 = prd_face_bgr * hist_mask_a + white
                    hist_match_1[hist_match_1 > 1.0] = 1.0

                    hist_match_2 = dst_face_bgr * hist_mask_a + white
                    hist_match_2[hist_match_1 > 1.0] = 1.0

                    prd_face_bgr = imagelib.color_hist_match(hist_match_1, hist_match_2, cfg.hist_match_threshold).astype(dtype=np.float32)

                # 如果模式包含 'seamless'，则执行图像无缝克隆操作
                if 'seamless' in cfg.mode:
                    # 用于 cv2.seamlessClone 的蒙版
                    img_face_seamless_mask_a = None
                    for i in range(1, 10):
                        a = img_face_mask_a > i / 10.0
                        if len(np.argwhere(a)) == 0:
                            continue
                        img_face_seamless_mask_a = img_face_mask_a.copy()
                        img_face_seamless_mask_a[a] = 1.0
                        img_face_seamless_mask_a[img_face_seamless_mask_a <= i / 10.0] = 0.0
                        break

                # 使用 cv2.seamlessClone 将合成的人脸融合到原始图像中
                out_img = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                out_img = np.clip(out_img, 0.0, 1.0)
                if 'seamless' in cfg.mode:
                    try:
                        # 计算与 cv2.seamlessClone 中相同的边界矩形和中心点，以防止抖动（不闪烁）
                        l, t, w, h = cv2.boundingRect((img_face_seamless_mask_a * 255).astype(np.uint8))
                        s_maskx, s_masky = int(l + w / 2), int(t + h / 2)
                        out_img = cv2.seamlessClone((out_img * 255).astype(np.uint8), img_bgr_uint8, (img_face_seamless_mask_a * 255).astype(np.uint8), (s_maskx, s_masky), cv2.NORMAL_CLONE)
                        out_img = out_img.astype(dtype=np.float32) / 255.0
                    except Exception as e:
                        # 在某些情况下，seamlessClone 可能失败
                        e_str = traceback.format_exc()

                        if 'MemoryError' in e_str:
                            raise Exception("Seamless 失败: " + e_str)  # 重新引发 MemoryError 以便由其他进程重新处理此数据
                        else:
                            print("Seamless 失败: " + e_str)

                cfg_mp = cfg.motion_blur_power / 100.0

                # 将合成的人脸与原始图像融合
                out_img = img_bgr * (1 - img_face_mask_a) + (out_img * img_face_mask_a)

                # 返回合成结果

                # 如果满足以下任一条件，将对生成的合成人脸进行额外处理
                if ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
                    cfg.mode == 'seamless-hist-match' or \
                    cfg_mp != 0 or \
                    cfg.blursharpen_amount != 0 or \
                    cfg.image_denoise_power != 0 or \
                    cfg.bicubic_degrade_power != 0:

                    # 使用仿射变换将合成的人脸图像放大到指定输出大小
                    out_face_bgr = cv2.warpAffine(out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)

                    # 如果模式中包含 'seamless' 并且颜色转移模式不为零，执行颜色转移
                    if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                        # 根据颜色转移模式进行颜色转移
                        if cfg.color_transfer_mode == 1:
                            out_face_bgr = imagelib.reinhard_color_transfer(out_face_bgr, dst_face_bgr, target_mask=wrk_face_mask_area_a, source_mask=wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 2:  # lct
                            out_face_bgr = imagelib.linear_color_transfer(out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 3:  # mkl
                            out_face_bgr = imagelib.color_transfer_mkl(out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 4:  # mkl-m
                            out_face_bgr = imagelib.color_transfer_mkl(out_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 5:  # idt
                            out_face_bgr = imagelib.color_transfer_idt(out_face_bgr, dst_face_bgr)
                        elif cfg.color_transfer_mode == 6:  # idt-m
                            out_face_bgr = imagelib.color_transfer_idt(out_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)
                        elif cfg.color_transfer_mode == 7:  # sot-m
                            out_face_bgr = imagelib.color_transfer_sot(out_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a, steps=10, batch_size=30)
                            out_face_bgr = np.clip(out_face_bgr, 0.0, 1.0)
                        elif cfg.color_transfer_mode == 8:  # mix-m
                            out_face_bgr = imagelib.color_transfer_mix(out_face_bgr * wrk_face_mask_area_a, dst_face_bgr * wrk_face_mask_area_a)

                    # 如果模式是 'seamless-hist-match'，执行直方图匹配
                    if cfg.mode == 'seamless-hist-match':
                        out_face_bgr = imagelib.color_hist_match(out_face_bgr, dst_face_bgr, cfg.hist_match_threshold)

                    # 如果 cfg_mp 不为零，执行运动模糊
                    if cfg_mp != 0:
                        k_size = int(frame_info.motion_power * cfg_mp)
                        if k_size >= 1:
                            k_size = np.clip(k_size + 1, 2, 50)
                            if cfg.super_resolution_power != 0:
                                k_size *= 2
                            out_face_bgr = imagelib.LinearMotionBlur(out_face_bgr, k_size, frame_info.motion_deg)

                    # 如果 cfg.blursharpen_amount 不为零，执行模糊或锐化处理
                    if cfg.blursharpen_amount != 0:
                        out_face_bgr = imagelib.blursharpen(out_face_bgr, cfg.sharpen_mode, 3, cfg.blursharpen_amount)

                    # 如果 cfg.image_denoise_power 不为零，执行图像去噪处理
                    if cfg.image_denoise_power != 0:
                        n = cfg.image_denoise_power
                        while n > 0:
                            img_bgr_denoised = cv2.medianBlur(img_bgr, 5)
                            if int(n / 100) != 0:
                                img_bgr = img_bgr_denoised
                            else:
                                pass_power = (n % 100) / 100.0
                                img_bgr = img_bgr * (1.0 - pass_power) + img_bgr_denoised * pass_power
                            n = max(n - 10, 0)

                    # 如果 cfg.bicubic_degrade_power 不为零，执行双三次降级和放大处理
                    if cfg.bicubic_degrade_power != 0:
                        p = 1.0 - cfg.bicubic_degrade_power / 101.0
                        img_bgr_downscaled = cv2.resize(img_bgr, (int(img_size[0] * p), int(img_size[1] * p)), interpolation=cv2.INTER_CUBIC)
                        img_bgr = cv2.resize(img_bgr_downscaled, img_size, interpolation=cv2.INTER_CUBIC)

                    # 使用仿射变换将处理后的人脸图像还原到原始图像尺寸
                    new_out = cv2.warpAffine(out_face_bgr, face_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

                    # 将处理后的人脸与原始图像融合
                    out_img = np.clip(img_bgr * (1 - img_face_mask_a) + (new_out * img_face_mask_a), 0, 1.0)

                # 如果 cfg.color_degrade_power 不为零，执行颜色降级处理
                if cfg.color_degrade_power != 0:
                    out_img_reduced = imagelib.reduce_colors(out_img, 256)
                    if cfg.color_degrade_power == 100:
                        out_img = out_img_reduced
                    else:
                        alpha = cfg.color_degrade_power / 100.0
                        out_img = (out_img * (1.0 - alpha) + out_img_reduced * alpha)
        out_merging_mask_a = img_face_mask_a

    # 如果 out_img 为空，将其设置为原始图像的副本
    if out_img is None:
        out_img = img_bgr.copy()

    # 返回最终的合成结果和合成蒙版
    return out_img, out_merging_mask_a

def MergeMasked(predictor_func,
                predictor_input_shape,
                face_enhancer_func,
                xseg_256_extract_func,
                cfg,
                frame_info):
    # 读取输入的图像并进行预处理
    img_bgr_uint8 = cv2_imread(frame_info.filepath)  # 以uint8格式读取图像
    img_bgr_uint8 = imagelib.normalize_channels(img_bgr_uint8, 3)  # 标准化通道值，确保通道值在0到255之间
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0  # 转换图像为浮点数格式，并将像素值缩放到0到1之间

    outs = []
    # 遍历每个检测到的人脸，并合并它们
    for face_num, img_landmarks in enumerate(frame_info.landmarks_list):
        # 调用MergeMaskedFace函数合并单个人脸
        out_img, out_img_merging_mask = MergeMaskedFace(predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, cfg, frame_info, img_bgr_uint8, img_bgr, img_landmarks)
        outs += [(out_img, out_img_merging_mask)]

    # 合并多个人脸输出
    final_img = None
    final_mask = None
    for img, merging_mask in outs:
        h, w, c = img.shape

        if final_img is None:
            final_img = img
            final_mask = merging_mask
        else:
            # 将多个人脸输出进行融合，加权叠加
            final_img = final_img * (1 - merging_mask) + img * merging_mask
            final_mask = np.clip(final_mask + merging_mask, 0, 1)

    # 将融合后的图像与融合蒙版连接起来
    final_img = np.concatenate([final_img, final_mask], -1)

    # 将最终的图像转换回uint8格式，并返回
    return (final_img * 255).astype(np.uint8)

