import numpy as np
import copy

from facelib import FaceType  # 导入 FaceType
from core.interact import interact as io  # 导入交互模块

class MergerConfig(object):
    # 定义合并配置的类，包括各种属性和方法

    TYPE_NONE = 0
    TYPE_MASKED = 1
    TYPE_FACE_AVATAR = 2
    TYPE_IMAGE = 3
    TYPE_IMAGE_WITH_LANDMARKS = 4

    def __init__(self, type=0, sharpen_mode=0, blursharpen_amount=0, **kwargs):
        # 初始化合并配置对象
        self.type = type
        self.sharpen_dict = {0: "None", 1: 'box', 2: 'gaussian'}

        # 可更改参数的默认值
        self.sharpen_mode = sharpen_mode
        self.blursharpen_amount = blursharpen_amount

    def copy(self):
        # 创建配置对象的副本
        return copy.copy(self)

    def ask_settings(self):
        # 询问用户配置参数的设置
        s = """选择锐化模式：\n"""
        for key in self.sharpen_dict.keys():
            s += f"""({key}) {self.sharpen_dict[key]}\n"""
        io.log_info(s)
        self.sharpen_mode = io.input_int("", 0, valid_list=self.sharpen_dict.keys(), help_message="通过应用锐化滤镜增强细节。")

        if self.sharpen_mode != 0:
            self.blursharpen_amount = np.clip(io.input_int("选择模糊/锐化程度", 0, add_info="-100..100"), -100, 100)

    def toggle_sharpen_mode(self):
        # 切换锐化模式
        a = list(self.sharpen_dict.keys())
        self.sharpen_mode = a[(a.index(self.sharpen_mode) + 1) % len(a)]

    def add_blursharpen_amount(self, diff):
        # 调整模糊/锐化程度
        self.blursharpen_amount = np.clip(self.blursharpen_amount + diff, -100, 100)

    def get_config(self):
        # 获取配置参数
        d = self.__dict__.copy()
        d.pop('type')
        return d

    def __eq__(self, other):
        # 检查可更改参数的相等性

        if isinstance(other, MergerConfig):
            return self.sharpen_mode == other.sharpen_mode and \
                   self.blursharpen_amount == other.blursharpen_amount

        return False

    def to_string(self, filename):
        # 将配置参数转换为字符串
        r = ""
        r += f"锐化模式：{self.sharpen_dict[self.sharpen_mode]}\n"
        r += f"模糊/锐化程度：{self.blursharpen_amount}\n"
        return r

# 定义一些字典
mode_dict = {0: 'original',
             1: 'overlay',
             2: 'hist-match',
             3: 'seamless',
             4: 'seamless-hist-match',
             5: 'raw-rgb',
             6: 'raw-predict'}

mode_str_dict = {mode_dict[key]: key for key in mode_dict.keys()}

mask_mode_dict = {0: 'full',
                  1: 'dst',
                  2: 'learned-prd',
                  3: 'learned-dst',
                  4: 'learned-prd*learned-dst',
                  5: 'learned-prd+learned-dst',
                  6: 'XSeg-prd',
                  7: 'XSeg-dst',
                  8: 'XSeg-prd*XSeg-dst',
                  9: 'learned-prd*learned-dst*XSeg-prd*XSeg-dst'
                  }
ctm_dict = {0: "None", 1: "rct", 2: "lct", 3: "mkl", 4: "mkl-m", 5: "idt", 6: "idt-m", 7: "sot-m", 8: "mix-m"}
ctm_str_dict = {None: 0, "rct": 1, "lct": 2, "mkl": 3, "mkl-m": 4, "idt": 5, "idt-m": 6, "sot-m": 7, "mix-m": 8}

# 定义合并配置的子类 MergerConfigMasked
class MergerConfigMasked(MergerConfig):
    def __init__(self, face_type=FaceType.FULL, default_mode='overlay', mode='overlay', masked_hist_match=True,
                 hist_match_threshold=238, mask_mode=4, erode_mask_modifier=0, blur_mask_modifier=0,
                 motion_blur_power=0, output_face_scale=0, super_resolution_power=0, color_transfer_mode=ctm_str_dict['rct'],
                 image_denoise_power=0, bicubic_degrade_power=0, color_degrade_power=0, **kwargs):

        # 调用父类 MergerConfig 的构造函数
        super().__init__(type=MergerConfig.TYPE_MASKED, **kwargs)

        # 初始化 MergerConfigMasked 特有的属性
        self.face_type = face_type
        if self.face_type not in [FaceType.HALF, FaceType.MID_FULL, FaceType.FULL, FaceType.WHOLE_FACE, FaceType.HEAD]:
            raise ValueError("MergerConfigMasked 不支持此类型的脸部。")

        self.default_mode = default_mode

        # 默认可更改参数的初始值
        if mode not in mode_str_dict:
            mode = mode_dict[1]

        self.mode = mode
        self.masked_hist_match = masked_hist_match
        self.hist_match_threshold = hist_match_threshold
        self.mask_mode = mask_mode
        self.erode_mask_modifier = erode_mask_modifier
        self.blur_mask_modifier = blur_mask_modifier
        self.motion_blur_power = motion_blur_power
        self.output_face_scale = output_face_scale
        self.super_resolution_power = super_resolution_power
        self.color_transfer_mode = color_transfer_mode
        self.image_denoise_power = image_denoise_power
        self.bicubic_degrade_power = bicubic_degrade_power
        self.color_degrade_power = color_degrade_power

    def copy(self):
        # 创建配置对象的副本
        return copy.copy(self)

    def set_mode(self, mode):
        # 设置合并模式
        self.mode = mode_dict.get(mode, self.default_mode)

    def toggle_masked_hist_match(self):
        # 切换是否进行掩码直方图匹配
        if self.mode == 'hist-match':
            self.masked_hist_match = not self.masked_hist_match

    def add_hist_match_threshold(self, diff):
        # 调整直方图匹配的阈值
        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            self.hist_match_threshold = np.clip(self.hist_match_threshold + diff, 0, 255)

    def toggle_mask_mode(self):
        # 切换掩码模式
        a = list(mask_mode_dict.keys())
        self.mask_mode = a[(a.index(self.mask_mode) + 1) % len(a)]

    def add_erode_mask_modifier(self, diff):
        # 调整腐蚀掩码的修正值
        self.erode_mask_modifier = np.clip(self.erode_mask_modifier + diff, -400, 400)

    def add_blur_mask_modifier(self, diff):
        # 调整模糊掩码的修正值
        self.blur_mask_modifier = np.clip(self.blur_mask_modifier + diff, 0, 400)

    def add_motion_blur_power(self, diff):
        # 调整运动模糊的强度
        self.motion_blur_power = np.clip(self.motion_blur_power + diff, 0, 100)

    def add_output_face_scale(self, diff):
        # 调整输出脸部的缩放比例
        self.output_face_scale = np.clip(self.output_face_scale + diff, -50, 50)

    def toggle_color_transfer_mode(self):
        # 切换颜色转移模式
        self.color_transfer_mode = (self.color_transfer_mode + 1) % (max(ctm_dict.keys()) + 1)

    def add_super_resolution_power(self, diff):
        # 调整超分辨率强度
        self.super_resolution_power = np.clip(self.super_resolution_power + diff, 0, 100)

    def add_color_degrade_power(self, diff):
        # 调整颜色降级强度
        self.color_degrade_power = np.clip(self.color_degrade_power + diff, 0, 100)

    def add_image_denoise_power(self, diff):
        # 调整图像降噪强度
        self.image_denoise_power = np.clip(self.image_denoise_power + diff, 0, 500)

    def add_bicubic_degrade_power(self, diff):
        # 调整双三次降级强度
        self.bicubic_degrade_power = np.clip(self.bicubic_degrade_power + diff, 0, 100)

    def ask_settings(self):
        # 询问用户配置参数的设置
        s = """选择模式：\n"""
        for key in mode_dict.keys():
            s += f"""({key}) {mode_dict[key]}\n"""
        io.log_info(s)
        mode = io.input_int("", mode_str_dict.get(self.default_mode, 1))

        self.mode = mode_dict.get(mode, self.default_mode)

        if 'raw' not in self.mode:
            if self.mode == 'hist-match':
                self.masked_hist_match = io.input_bool("是否进行掩码直方图匹配？", True)

            if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
                self.hist_match_threshold = np.clip(io.input_int("设置直方图匹配阈值", 255, add_info="0..255"), 0, 255)

        s = """选择掩码模式：\n"""
        for key in mask_mode_dict.keys():
            s += f"""({key}) {mask_mode_dict[key]}\n"""
        io.log_info(s)
        self.mask_mode = io.input_int("", 1, valid_list=mask_mode_dict.keys())

        if 'raw' not in self.mode:
            self.erode_mask_modifier = np.clip(io.input_int("选择腐蚀掩码修正值", 0, add_info="-400..400"), -400, 400)
            self.blur_mask_modifier = np.clip(io.input_int("选择模糊掩码修正值", 0, add_info="0..400"), 0, 400)
            self.motion_blur_power = np.clip(io.input_int("选择运动模糊强度", 0, add_info="0..100"), 0, 100)

        self.output_face_scale = np.clip(io.input_int("选择输出脸部缩放比例", 0, add_info="-50..50"), -50, 50)

        if 'raw' not in self.mode:
            self.color_transfer_mode = io.input_str("颜色转移到预测脸部", None, valid_list=list(ctm_str_dict.keys())[1:])
            self.color_transfer_mode = ctm_str_dict[self.color_transfer_mode]

        super().ask_settings()

        self.super_resolution_power = np.clip(
            io.input_int("选择超分辨率强度", 0, add_info="0..100", help_message="通过应用超分辨率网络来增强细节。"), 0, 100)

        if 'raw' not in self.mode:
            self.image_denoise_power = np.clip(io.input_int("选择图像降级去噪强度", 0, add_info="0..500"), 0, 500)
            self.bicubic_degrade_power = np.clip(io.input_int("选择图像降级双三次缩放强度", 0, add_info="0..100"), 0, 100)
            self.color_degrade_power = np.clip(io.input_int("最终图像颜色降级强度", 0, add_info="0..100"), 0, 100)

        io.log_info("")

    def __eq__(self, other):
        # 检查可更改参数的相等性

        if isinstance(other, MergerConfigMasked):
            return super().__eq__(other) and \
                   self.mode == other.mode and \
                   self.masked_hist_match == other.masked_hist_match and \
                   self.hist_match_threshold == other.hist_match_threshold and \
                   self.mask_mode == other.mask_mode and \
                   self.erode_mask_modifier == other.erode_mask_modifier and \
                   self.blur_mask_modifier == other.blur_mask_modifier and \
                   self.motion_blur_power == other.motion_blur_power and \
                   self.output_face_scale == other.output_face_scale and \
                   self.color_transfer_mode == other.color_transfer_mode and \
                   self.super_resolution_power == other.super_resolution_power and \
                   self.image_denoise_power == other.image_denoise_power and \
                   self.bicubic_degrade_power == other.bicubic_degrade_power and \
                   self.color_degrade_power == other.color_degrade_power

        return False

    def to_string(self, filename):
        r = (
            f"""MergerConfig {filename}:\n"""
            f"""模式：{self.mode}\n"""
            )

        if self.mode == 'hist-match':
            r += f"""是否进行掩码直方图匹配：{self.masked_hist_match}\n"""

        if self.mode == 'hist-match' or self.mode == 'seamless-hist-match':
            r += f"""直方图匹配阈值：{self.hist_match_threshold}\n"""

        r += f"""掩码模式：{mask_mode_dict[self.mask_mode]}\n"""

        if 'raw' not in self.mode:
            r += (f"""腐蚀掩码修正值：{self.erode_mask_modifier}\n"""
                  f"""模糊掩码修正值：{self.blur_mask_modifier}\n"""
                  f"""运动模糊强度：{self.motion_blur_power}\n""")

        r += f"""输出脸部缩放比例：{self.output_face_scale}\n"""

        if 'raw' not in self.mode:
            r += f"""颜色转移模式：{ctm_dict[self.color_transfer_mode]}\n"""
            r += super().to_string(filename)

        r += f"""超分辨率强度：{self.super_resolution_power}\n"""

        if 'raw' not in self.mode:
            r += (f"""图像降噪强度：{self.image_denoise_power}\n"""
                  f"""图像降级双三次缩放强度：{self.bicubic_degrade_power}\n"""
                  f"""最终图像颜色降级强度：{self.color_degrade_power}\n""")

        r += "================"

        return r

class MergerConfigFaceAvatar(MergerConfig):

    def __init__(self, temporal_face_count=0,
                       add_source_image=False):
        super().__init__(type=MergerConfig.TYPE_FACE_AVATAR)
        self.temporal_face_count = temporal_face_count

        # 可更改参数
        self.add_source_image = add_source_image

    def copy(self):
        return copy.copy(self)

    # 重写
    def ask_settings(self):
        # 询问是否添加源图像用于比较
        self.add_source_image = io.input_bool("是否添加源图像？", False, help_message="添加源图像以进行比较。")
        super().ask_settings()

    def toggle_add_source_image(self):
        # 切换是否添加源图像选项
        self.add_source_image = not self.add_source_image

    # 重写
    def __eq__(self, other):
        # 检查可更改参数的相等性

        if isinstance(other, MergerConfigFaceAvatar):
            return super().__eq__(other) and \
                   self.add_source_image == other.add_source_image

        return False

    # 重写
    def to_string(self, filename):
        return (f"MergerConfig {filename}:\n"
                f"add_source_image : {self.add_source_image}\n") + \
                super().to_string(filename) + "================"
