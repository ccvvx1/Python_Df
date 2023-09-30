import multiprocessing
import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from merger import MergeFaceAvatar, MergeMasked, MergerConfig

from .MergerScreen import Screen, ScreenManager

# MERGER_DEBUG 用于控制调试模式
MERGER_DEBUG = False

class InteractiveMergerSubprocessor(Subprocessor):

    # Frame 类用于表示帧信息
    class Frame(object):
        def __init__(self, prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None):
            # 初始化帧信息对象
            # prev_temporal_frame_infos: 前一时刻帧信息列表
            # frame_info: 当前帧信息
            # next_temporal_frame_infos: 下一时刻帧信息列表
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = None
            self.output_mask_filepath = None

            self.idx = None
            self.cfg = None
            self.is_done = False
            self.is_processing = False
            self.is_shown = False
            self.image = None

    # ProcessingFrame 类用于表示正在处理的帧信息
    class ProcessingFrame(object):
        def __init__(self, idx=None,
                           cfg=None,
                           prev_temporal_frame_infos=None,
                           frame_info=None,
                           next_temporal_frame_infos=None,
                           output_filepath=None,
                           output_mask_filepath=None,
                           need_return_image=False):
            # 初始化正在处理的帧信息对象
            # idx: 帧的索引
            # cfg: 配置信息
            # prev_temporal_frame_infos: 前一时刻帧信息列表
            # frame_info: 当前帧信息
            # next_temporal_frame_infos: 下一时刻帧信息列表
            # output_filepath: 输出文件路径
            # output_mask_filepath: 输出遮罩文件路径
            # need_return_image: 是否需要返回图像
            self.idx = idx
            self.cfg = cfg
            self.prev_temporal_frame_infos = prev_temporal_frame_infos
            self.frame_info = frame_info
            self.next_temporal_frame_infos = next_temporal_frame_infos
            self.output_filepath = output_filepath
            self.output_mask_filepath = output_mask_filepath

            self.need_return_image = need_return_image
            if self.need_return_image:
                self.image = None

    # Cli 类继承自 Subprocessor.Cli，用于与子进程进行通信
    class Cli(Subprocessor.Cli):

        # 初始化方法
        #override
        def on_initialize(self, client_dict):
            # 在子进程中进行初始化
            self.log_info ('Running on %s.' % (client_dict['device_name']) )
            self.device_idx  = client_dict['device_idx']
            self.device_name = client_dict['device_name']
            self.predictor_func = client_dict['predictor_func']
            self.predictor_input_shape = client_dict['predictor_input_shape']
            self.face_enhancer_func = client_dict['face_enhancer_func']
            self.xseg_256_extract_func = client_dict['xseg_256_extract_func']

            # 将 stdin 转移并设置为工作中的代码.interact，以便在调试子进程中使用
            stdin_fd = client_dict['stdin_fd']
            if stdin_fd is not None:
                sys.stdin = os.fdopen(stdin_fd)

            return None  # 返回空值

        #override
        def process_data(self, pf):  # pf=ProcessingFrame
            # 处理数据的方法，对图像进行处理
            cfg = pf.cfg.copy()

            frame_info = pf.frame_info
            filepath = frame_info.filepath

            if len(frame_info.landmarks_list) == 0:
                # 如果未检测到人脸关键点
                if cfg.mode == 'raw-predict':
                    h, w, c = self.predictor_input_shape
                    img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
                    img_mask = np.zeros((h, w, 1), dtype=np.uint8)
                else:
                    self.log_info(f'{filepath.name} 未检测到人脸，将图像复制并保留原图')
                    img_bgr = cv2_imread(filepath)
                    imagelib.normalize_channels(img_bgr, 3)
                    h, w, c = img_bgr.shape
                    img_mask = np.zeros((h, w, 1), dtype=img_bgr.dtype)

                # 保存处理后的图像和掩码
                cv2_imwrite(pf.output_filepath, img_bgr)
                cv2_imwrite(pf.output_mask_filepath, img_mask)

                if pf.need_return_image:
                    pf.image = np.concatenate([img_bgr, img_mask], axis=-1)

            else:
                if cfg.type == MergerConfig.TYPE_MASKED:
                    try:
                        final_img = MergeMasked(self.predictor_func, self.predictor_input_shape,
                                                face_enhancer_func=self.face_enhancer_func,
                                                xseg_256_extract_func=self.xseg_256_extract_func,
                                                cfg=cfg,
                                                frame_info=frame_info)
                    except Exception as e:
                        e_str = traceback.format_exc()
                        if 'MemoryError' in e_str:
                            raise Subprocessor.SilenceException
                        else:
                            raise Exception(f'合成文件时出错 [{filepath}]: {e_str}')

                elif cfg.type == MergerConfig.TYPE_FACE_AVATAR:
                    final_img = MergeFaceAvatar(self.predictor_func, self.predictor_input_shape,
                                                cfg, pf.prev_temporal_frame_infos,
                                                pf.frame_info,
                                                pf.next_temporal_frame_infos)

                # 保存合成后的图像和掩码
                cv2_imwrite(pf.output_filepath, final_img[..., 0:3])
                cv2_imwrite(pf.output_mask_filepath, final_img[..., 3:4])

                if pf.need_return_image:
                    pf.image = final_img

            return pf

        #overridable
        def get_data_name(self, pf):
            # 返回数据的标识符，通常为文件路径
            return pf.frame_info.filepath





    #override
    def __init__(self, is_interactive, merger_session_filepath, predictor_func, predictor_input_shape, face_enhancer_func, xseg_256_extract_func, merger_config, frames, frames_root_path, output_path, output_mask_path, model_iter, subprocess_count=4):
        # 检查帧列表是否为空，如果为空则引发异常
        if len(frames) == 0:
            raise ValueError("len(frames) == 0")

        # 调用父类的构造函数以初始化子进程
        super().__init__('Merger', InteractiveMergerSubprocessor.Cli, io_loop_sleep_time=0.001)

        # 存储传递进来的参数
        self.is_interactive = is_interactive  # 表示是否以交互模式运行
        self.merger_session_filepath = Path(merger_session_filepath)  # 合并会话文件的路径
        self.merger_config = merger_config  # 合并配置对象

        self.predictor_func = predictor_func  # 预测函数，用于合并过程
        self.predictor_input_shape = predictor_input_shape  # 输入形状，用于预测函数

        self.face_enhancer_func = face_enhancer_func  # 人脸增强函数，用于合并过程
        self.xseg_256_extract_func = xseg_256_extract_func  # 用于提取 XSeg 256 模型的函数

        self.frames_root_path = frames_root_path  # 帧图像的根路径
        self.output_path = output_path  # 输出合并图像的路径
        self.output_mask_path = output_mask_path  # 输出合并掩码图像的路径
        self.model_iter = model_iter  # 模型的迭代次数

        self.prefetch_frame_count = self.process_count = subprocess_count  # 预取和处理的帧数

        session_data = None

        # 如果以交互模式运行并且合并会话文件存在，尝试加载之前的会话数据
        if self.is_interactive and self.merger_session_filepath.exists():
            io.input_skip_pending()
            
            # 提示用户是否要使用已保存的会话数据
            if io.input_bool("使用已保存的会话？", True):
                try:
                    # 尝试从文件中加载会话数据
                    with open(str(self.merger_session_filepath), "rb") as f:
                        session_data = pickle.loads(f.read())

                except Exception as e:
                    pass

        rewind_to_frame_idx = None
        self.frames = frames  # 帧的列表
        self.frames_idxs = [*range(len(self.frames))]  # 帧的索引列表，表示待处理的帧
        self.frames_done_idxs = []  # 已处理完成的帧的索引列表

        if self.is_interactive and session_data is not None:
            # 如果以交互模式运行且存在已保存的会话数据

            # 尝试从会话数据中获取各种信息
            s_frames = session_data.get('frames', None)  # 获取已保存的帧列表
            s_frames_idxs = session_data.get('frames_idxs', None)  # 获取已保存的帧索引列表
            s_frames_done_idxs = session_data.get('frames_done_idxs', None)  # 获取已保存的已完成帧的索引列表
            s_model_iter = session_data.get('model_iter', None)  # 获取已保存的模型迭代次数

            # 检查加载的会话数据是否匹配当前任务
            frames_equal = (s_frames is not None) and \
                        (s_frames_idxs is not None) and \
                        (s_frames_done_idxs is not None) and \
                        (s_model_iter is not None) and \
                        (len(frames) == len(s_frames))  # 帧的数量必须匹配

            if frames_equal:
                for i in range(len(frames)):
                    frame = frames[i]
                    s_frame = s_frames[i]

                    # 检查帧的文件名是否匹配
                    if frame.frame_info.filepath.name != s_frame.frame_info.filepath.name:
                        frames_equal = False
                    if not frames_equal:
                        break

            if frames_equal:
                # 如果会话数据匹配当前任务

                # 输出日志，表示正在使用已保存的会话数据
                io.log_info ('正在使用保存的会话数据：' + '/'.join (self.merger_session_filepath.parts[-2:]) )

                for frame in s_frames:
                    if frame.cfg is not None:
                        # 重新创建 MergerConfig 类，使用具有 get_config() 作为字典参数的构造函数
                        # 这样，如果添加了任何新的参数，旧的合并会话仍将正常工作
                        frame.cfg = frame.cfg.__class__( **frame.cfg.get_config() )

                self.frames = s_frames
                self.frames_idxs = s_frames_idxs
                self.frames_done_idxs = s_frames_done_idxs

                if self.model_iter != s_model_iter:
                    # 模型更多地训练过，需要重新计算所有帧
                    rewind_to_frame_idx = -1
                    for frame in self.frames:
                        frame.is_done = False
                elif len(self.frames_idxs) == 0:
                    # 所有帧都已完成？
                    rewind_to_frame_idx = -1

                if len(self.frames_idxs) != 0:
                    cur_frame = self.frames[self.frames_idxs[0]]
                    cur_frame.is_shown = False

            if not frames_equal:
                session_data = None

        if session_data is None:
            # 如果没有匹配的会话数据或需要重新计算帧

            # 删除输出路径中的所有图像文件
            for filename in pathex.get_image_paths(self.output_path):
                Path(filename).unlink()

            # 删除输出掩码路径中的所有图像文件
            for filename in pathex.get_image_paths(self.output_mask_path):
                Path(filename).unlink()

            # 复制合并配置到第一帧
            frames[0].cfg = self.merger_config.copy()

        for i in range(len(self.frames)):
            frame = self.frames[i]
            frame.idx = i
            frame.output_filepath      = self.output_path      / (frame.frame_info.filepath.stem + '.png')
            frame.output_mask_filepath = self.output_mask_path / (frame.frame_info.filepath.stem + '.png')

            if not frame.output_filepath.exists() or \
            not frame.output_mask_filepath.exists():
                # 如果某些帧不存在，需要重新计算并回退
                frame.is_done = False
                frame.is_shown = False

                if rewind_to_frame_idx is None:
                    rewind_to_frame_idx = i-1
                else:
                    rewind_to_frame_idx = min(rewind_to_frame_idx, i-1)

        if rewind_to_frame_idx is not None:
            while len(self.frames_done_idxs) > 0:
                if self.frames_done_idxs[-1] > rewind_to_frame_idx:
                    prev_frame = self.frames[self.frames_done_idxs.pop()]
                    self.frames_idxs.insert(0, prev_frame.idx)
                else:
                    break

    # 生成处理信息的生成器
    #override
    def process_info_generator(self):
        # 如果 MERGER_DEBUG 为真，则只使用 CPU 0 处理
        r = [0] if MERGER_DEBUG else range(self.process_count)

        for i in r:
            # 为每个处理器生成处理信息
            yield 'CPU%d' % (i), {}, {
                'device_idx': i,
                'device_name': 'CPU%d' % (i),
                'predictor_func': self.predictor_func,
                'predictor_input_shape': self.predictor_input_shape,
                'face_enhancer_func': self.face_enhancer_func,
                'xseg_256_extract_func': self.xseg_256_extract_func,
                'stdin_fd': sys.stdin.fileno() if MERGER_DEBUG else None
            }

    # 在客户端初始化完成后调用的函数
    def on_clients_initialized(self):
        # 显示合并进度条
        io.progress_bar("合并中", len(self.frames_idxs) + len(self.frames_done_idxs), initial=len(self.frames_done_idxs))

        # 设置一些状态标志
        self.process_remain_frames = not self.is_interactive
        self.is_interactive_quitting = not self.is_interactive

        if self.is_interactive:
            # 加载合并帮助图像
            help_images = {
                MergerConfig.TYPE_MASKED: cv2_imread(str(Path(__file__).parent / 'gfx' / 'help_merger_masked.jpg')),
                MergerConfig.TYPE_FACE_AVATAR: cv2_imread(str(Path(__file__).parent / 'gfx' / 'help_merger_face_avatar.jpg')),
            }

            # 创建主屏幕和帮助屏幕
            self.main_screen = Screen(initial_scale_to_width=1368, image=None, waiting_icon=True)
            self.help_screen = Screen(initial_scale_to_height=768, image=help_images[self.merger_config.type], waiting_icon=False)
            self.screen_manager = ScreenManager("合并", [self.main_screen, self.help_screen], capture_keys=True)
            self.screen_manager.set_current(self.help_screen)
            self.screen_manager.show_current()

            # 定义按键映射函数
            self.masked_keys_funcs = {
                '`': lambda cfg, shift_pressed: cfg.set_mode(0),
                '1': lambda cfg, shift_pressed: cfg.set_mode(1),
                '2': lambda cfg, shift_pressed: cfg.set_mode(2),
                '3': lambda cfg, shift_pressed: cfg.set_mode(3),
                '4': lambda cfg, shift_pressed: cfg.set_mode(4),
                '5': lambda cfg, shift_pressed: cfg.set_mode(5),
                '6': lambda cfg, shift_pressed: cfg.set_mode(6),
                'q': lambda cfg, shift_pressed: cfg.add_hist_match_threshold(1 if not shift_pressed else 5),
                'a': lambda cfg, shift_pressed: cfg.add_hist_match_threshold(-1 if not shift_pressed else -5),
                'w': lambda cfg, shift_pressed: cfg.add_erode_mask_modifier(1 if not shift_pressed else 5),
                's': lambda cfg, shift_pressed: cfg.add_erode_mask_modifier(-1 if not shift_pressed else -5),
                'e': lambda cfg, shift_pressed: cfg.add_blur_mask_modifier(1 if not shift_pressed else 5),
                'd': lambda cfg, shift_pressed: cfg.add_blur_mask_modifier(-1 if not shift_pressed else -5),
                'r': lambda cfg, shift_pressed: cfg.add_motion_blur_power(1 if not shift_pressed else 5),
                'f': lambda cfg, shift_pressed: cfg.add_motion_blur_power(-1 if not shift_pressed else -5),
                't': lambda cfg, shift_pressed: cfg.add_super_resolution_power(1 if not shift_pressed else 5),
                'g': lambda cfg, shift_pressed: cfg.add_super_resolution_power(-1 if not shift_pressed else -5),
                'y': lambda cfg, shift_pressed: cfg.add_blursharpen_amount(1 if not shift_pressed else 5),
                'h': lambda cfg, shift_pressed: cfg.add_blursharpen_amount(-1 if not shift_pressed else -5),
                'u': lambda cfg, shift_pressed: cfg.add_output_face_scale(1 if not shift_pressed else 5),
                'j': lambda cfg, shift_pressed: cfg.add_output_face_scale(-1 if not shift_pressed else -5),
                'i': lambda cfg, shift_pressed: cfg.add_image_denoise_power(1 if not shift_pressed else 5),
                'k': lambda cfg, shift_pressed: cfg.add_image_denoise_power(-1 if not shift_pressed else -5),
                'o': lambda cfg, shift_pressed: cfg.add_bicubic_degrade_power(1 if not shift_pressed else 5),
                'l': lambda cfg, shift_pressed: cfg.add_bicubic_degrade_power(-1 if not shift_pressed else -5),
                'p': lambda cfg, shift_pressed: cfg.add_color_degrade_power(1 if not shift_pressed else 5),
                ';': lambda cfg, shift_pressed: cfg.add_color_degrade_power(-1),
                ':': lambda cfg, shift_pressed: cfg.add_color_degrade_power(-5),
                'z': lambda cfg, shift_pressed: cfg.toggle_masked_hist_match(),
                'x': lambda cfg, shift_pressed: cfg.toggle_mask_mode(),
                'c': lambda cfg, shift_pressed: cfg.toggle_color_transfer_mode(),
                'n': lambda cfg, shift_pressed: cfg.toggle_sharpen_mode(),
            }
            
            # 将按键映射函数的键存储为列表
            self.masked_keys = list(self.masked_keys_funcs.keys())

    # 在客户端最终化时调用的函数
    def on_clients_finalized(self):
        # 关闭进度条
        io.progress_bar_close()

        if self.is_interactive:
            # 最终化屏幕管理器
            self.screen_manager.finalize()

            # 清空帧的输出路径、遮罩输出路径和图像
            for frame in self.frames:
                frame.output_filepath = None
                frame.output_mask_filepath = None
                frame.image = None

            # 保存会话数据到文件
            session_data = {
                'frames': self.frames,
                'frames_idxs': self.frames_idxs,
                'frames_done_idxs': self.frames_done_idxs,
                'model_iter': self.model_iter,
            }
            self.merger_session_filepath.write_bytes(pickle.dumps(session_data))

            io.log_info("会话已保存到" + '/'.join(self.merger_session_filepath.parts[-2:]))

    # 在每个时钟周期调用的函数
    #override
    def on_tick(self):
        io.process_messages()

        go_prev_frame = False
        go_first_frame = False
        go_prev_frame_overriding_cfg = False
        go_first_frame_overriding_cfg = False

        go_next_frame = self.process_remain_frames
        go_next_frame_overriding_cfg = False
        go_last_frame_overriding_cfg = False

        cur_frame = None
        if len(self.frames_idxs) != 0:
            cur_frame = self.frames[self.frames_idxs[0]]

        if self.is_interactive:
            screen_image = None if self.process_remain_frames else \
                                self.main_screen.get_image()

            self.main_screen.set_waiting_icon(self.process_remain_frames or \
                                            self.is_interactive_quitting)

            if cur_frame is not None and not self.is_interactive_quitting:
                if not self.process_remain_frames:
                    if cur_frame.is_done:
                        if not cur_frame.is_shown:
                            if cur_frame.image is None:
                                image = cv2_imread(cur_frame.output_filepath, verbose=False)
                                image_mask = cv2_imread(cur_frame.output_mask_filepath, verbose=False)
                                if image is None or image_mask is None:
                                    # 无法读取？则重新计算
                                    cur_frame.is_done = False
                                else:
                                    image = imagelib.normalize_channels(image, 3)
                                    image_mask = imagelib.normalize_channels(image_mask, 1)
                                    cur_frame.image = np.concatenate([image, image_mask], -1)

                            if cur_frame.is_done:
                                io.log_info(cur_frame.cfg.to_string(cur_frame.frame_info.filepath.name))
                                cur_frame.is_shown = True
                                screen_image = cur_frame.image
                    else:
                        self.main_screen.set_waiting_icon(True)

            self.main_screen.set_image(screen_image)
            self.screen_manager.show_current()

            key_events = self.screen_manager.get_key_events()
            key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0, 0, False, False, False)

            if key == 9:  # Tab键
                self.screen_manager.switch_screens()
            else:
                if key == 27:  # Esc键
                    self.is_interactive_quitting = True
                elif self.screen_manager.get_current() is self.main_screen:

                    if self.merger_config.type == MergerConfig.TYPE_MASKED and chr_key in self.masked_keys:
                        self.process_remain_frames = False

                        if cur_frame is not None:
                            cfg = cur_frame.cfg
                            prev_cfg = cfg.copy()

                            if cfg.type == MergerConfig.TYPE_MASKED:
                                self.masked_keys_funcs[chr_key](cfg, shift_pressed)

                            if prev_cfg != cfg:
                                io.log_info(cfg.to_string(cur_frame.frame_info.filepath.name))
                                cur_frame.is_done = False
                                cur_frame.is_shown = False
                    else:

                        if chr_key == ',' or chr_key == 'm':
                            self.process_remain_frames = False
                            go_prev_frame = True

                            if chr_key == ',':
                                if shift_pressed:
                                    go_first_frame = True

                            elif chr_key == 'm':
                                if not shift_pressed:
                                    go_prev_frame_overriding_cfg = True
                                else:
                                    go_first_frame_overriding_cfg = True

                        elif chr_key == '.' or chr_key == '/':
                            self.process_remain_frames = False
                            go_next_frame = True

                            if chr_key == '.':
                                if shift_pressed:
                                    self.process_remain_frames = not self.process_remain_frames

                            elif chr_key == '/':
                                if not shift_pressed:
                                    go_next_frame_overriding_cfg = True
                                else:
                                    go_last_frame_overriding_cfg = True

                        elif chr_key == '-':
                            self.screen_manager.get_current().diff_scale(-0.1)
                        elif chr_key == '=':
                            self.screen_manager.get_current().diff_scale(0.1)
                        elif chr_key == 'v':
                            self.screen_manager.get_current().toggle_show_checker_board()

        # 如果要前进到前一帧
        if go_prev_frame:
            if cur_frame is None or cur_frame.is_done:
                if cur_frame is not None:
                    cur_frame.image = None

                while True:
                    if len(self.frames_done_idxs) > 0:
                        prev_frame = self.frames[self.frames_done_idxs.pop()]
                        self.frames_idxs.insert(0, prev_frame.idx)
                        prev_frame.is_shown = False
                        io.progress_bar_inc(-1)

                        if cur_frame is not None and (go_prev_frame_overriding_cfg or go_first_frame_overriding_cfg):
                            if prev_frame.cfg != cur_frame.cfg:
                                prev_frame.cfg = cur_frame.cfg.copy()
                                prev_frame.is_done = False

                        cur_frame = prev_frame

                    if go_first_frame_overriding_cfg or go_first_frame:
                        if len(self.frames_done_idxs) > 0:
                            continue
                    break

        # 如果要前进到下一帧
        elif go_next_frame:
            if cur_frame is not None and cur_frame.is_done:
                cur_frame.image = None
                cur_frame.is_shown = True
                self.frames_done_idxs.append(cur_frame.idx)
                self.frames_idxs.pop(0)
                io.progress_bar_inc(1)

                f = self.frames

                if len(self.frames_idxs) != 0:
                    next_frame = f[self.frames_idxs[0]]
                    next_frame.is_shown = False

                    if go_next_frame_overriding_cfg or go_last_frame_overriding_cfg:

                        if go_next_frame_overriding_cfg:
                            to_frames = next_frame.idx+1
                        else:
                            to_frames = len(f)

                        for i in range(next_frame.idx, to_frames):
                            f[i].cfg = None

                    for i in range(min(len(self.frames_idxs), self.prefetch_frame_count)):
                        frame = f[self.frames_idxs[i]]
                        if frame.cfg is None:
                            if i == 0:
                                frame.cfg = cur_frame.cfg.copy()
                            else:
                                frame.cfg = f[self.frames_idxs[i-1]].cfg.copy()

                            frame.is_done = False # 初始化重新计算
                            frame.is_shown = False

            if len(self.frames_idxs) == 0:
                self.process_remain_frames = False

        # 返回一个布尔值，表示是否继续处理下一帧
        return (self.is_interactive and self.is_interactive_quitting) or \
            (not self.is_interactive and self.process_remain_frames == False)

    # 在数据返回时调用的函数
    #override
    def on_data_return(self, host_dict, pf):
        frame = self.frames[pf.idx]
        frame.is_done = False
        frame.is_processing = False

    # 在结果返回时调用的函数
    #override
    def on_result(self, host_dict, pf_sent, pf_result):
        frame = self.frames[pf_result.idx]
        frame.is_processing = False
        if frame.cfg == pf_result.cfg:
            frame.is_done = True
            frame.image = pf_result.image

    # 获取要处理的数据
    #override
    def get_data(self, host_dict):
        if self.is_interactive and self.is_interactive_quitting:
            return None

        for i in range(min(len(self.frames_idxs), self.prefetch_frame_count)):
            frame = self.frames[self.frames_idxs[i]]

            if not frame.is_done and not frame.is_processing and frame.cfg is not None:
                frame.is_processing = True
                return InteractiveMergerSubprocessor.ProcessingFrame(
                    idx=frame.idx,
                    cfg=frame.cfg.copy(),
                    prev_temporal_frame_infos=frame.prev_temporal_frame_infos,
                    frame_info=frame.frame_info,
                    next_temporal_frame_infos=frame.next_temporal_frame_infos,
                    output_filepath=frame.output_filepath,
                    output_mask_filepath=frame.output_mask_filepath,
                    need_return_image=True
                )

        return None

    # 获取处理结果
    #override
    def get_result(self):
        return 0
