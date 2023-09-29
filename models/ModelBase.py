import colorsys
import inspect
import json
import multiprocessing
import operator
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from core import imagelib, pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from samplelib import SampleGeneratorBase

# 基础模型类，用于派生其他模型类
class ModelBase(object):
    def __init__(self, is_training=False,
                       is_exporting=False,
                       saved_models_path=None,
                       training_data_src_path=None,
                       training_data_dst_path=None,
                       pretraining_data_path=None,
                       pretrained_model_path=None,
                       no_preview=False,
                       force_model_name=None,
                       force_gpu_idxs=None,
                       cpu_only=False,
                       debug=False,
                       force_model_class_name=None,
                       silent_start=False,
                       **kwargs):
        self.is_training = is_training
        self.is_exporting = is_exporting
        self.saved_models_path = saved_models_path
        self.training_data_src_path = training_data_src_path
        self.training_data_dst_path = training_data_dst_path
        self.pretraining_data_path = pretraining_data_path
        self.pretrained_model_path = pretrained_model_path
        self.no_preview = no_preview
        self.debug = debug

        # 获取模型类名
        self.model_class_name = model_class_name = Path(inspect.getmodule(self).__file__).parent.name.rsplit("_", 1)[1]

        if force_model_class_name is None:
            if force_model_name is not None:
                self.model_name = force_model_name
            else:
                while True:
                    # 收集所有模型数据文件
                    saved_models_names = []
                    for filepath in pathex.get_file_paths(saved_models_path):
                        filepath_name = filepath.name
                        if filepath_name.endswith(f'{model_class_name}_data.dat'):
                            saved_models_names += [(filepath_name.split('_')[0], os.path.getmtime(filepath))]

                    # 按修改时间排序
                    saved_models_names = sorted(saved_models_names, key=operator.itemgetter(1), reverse=True)
                    saved_models_names = [x[0] for x in saved_models_names]

                    if len(saved_models_names) != 0:
                        if silent_start:
                            self.model_name = saved_models_names[0]
                            io.log_info(f'Silent start: choosed model "{self.model_name}"')
                        else:
                            io.log_info("选择一个已保存的模型，或输入名称创建新模型。")
                            io.log_info("[r] : 重命名")
                            io.log_info("[d] : 删除")
                            io.log_info("")
                            for i, model_name in enumerate(saved_models_names):
                                s = f"[{i}] : {model_name} "
                                if i == 0:
                                    s += "- 最新"
                                io.log_info(s)

                            inp = io.input_str(f"", "0", show_default_value=False)
                            model_idx = -1
                            try:
                                model_idx = np.clip(int(inp), 0, len(saved_models_names) - 1)
                            except:
                                pass

                            if model_idx == -1:
                                if len(inp) == 1:
                                    is_rename = inp[0] == 'r'
                                    is_delete = inp[0] == 'd'

                                    if is_rename or is_delete:
                                        if len(saved_models_names) != 0:

                                            if is_rename:
                                                name = io.input_str(f"输入要重命名的模型名称")
                                            elif is_delete:
                                                name = io.input_str(f"输入要删除的模型名称")

                                            if name in saved_models_names:

                                                if is_rename:
                                                    new_model_name = io.input_str(f"输入模型的新名称")

                                                for filepath in pathex.get_paths(saved_models_path):
                                                    filepath_name = filepath.name

                                                    model_filename, remain_filename = filepath_name.split('_', 1)
                                                    if model_filename == name:

                                                        if is_rename:
                                                            new_filepath = filepath.parent / (new_model_name + '_' + remain_filename)
                                                            filepath.rename(new_filepath)
                                                        elif is_delete:
                                                            filepath.unlink()
                                        continue

                                self.model_name = inp
                            else:
                                self.model_name = saved_models_names[model_idx]

                    else:
                        self.model_name = io.input_str(f"未找到已保存的模型。输入新模型的名称", "new")
                        self.model_name = self.model_name.replace('_', ' ')
                    break

            self.model_name = self.model_name + '_' + self.model_class_name
        else:
            self.model_name = force_model_class_name

        self.iter = 0
        self.options = {}
        self.options_show_override = {}
        self.loss_history = []
        self.sample_for_preview = None
        self.choosed_gpu_indexes = None



        model_data = {}
        self.model_data_path = Path(self.get_strpath_storage_for_file('data.dat'))
        if self.model_data_path.exists():
            io.log_info(f"加载 {self.model_name} 模型...")
            model_data = pickle.loads(self.model_data_path.read_bytes())
            self.iter = model_data.get('iter', 0)
            if self.iter != 0:
                self.options = model_data['options']
                self.loss_history = model_data.get('loss_history', [])
                self.sample_for_preview = model_data.get('sample_for_preview', None)
                self.choosed_gpu_indexes = model_data.get('choosed_gpu_indexes', None)

        if self.is_first_run():
            io.log_info("\n模型首次运行。")

        if silent_start:
            self.device_config = nn.DeviceConfig.BestGPU()
            io.log_info(f"静默启动：选择设备 {'CPU' if self.device_config.cpu_only else self.device_config.devices[0].name}")
        else:
            self.device_config = nn.DeviceConfig.GPUIndexes(force_gpu_idxs or nn.ask_choose_device_idxs(suggest_best_multi_gpu=True)) \
                                if not cpu_only else nn.DeviceConfig.CPU()

        nn.initialize(self.device_config)

        ####
        self.default_options_path = saved_models_path / f'{self.model_class_name}_default_options.dat'
        self.default_options = {}
        if self.default_options_path.exists():
            try:
                self.default_options = pickle.loads(self.default_options_path.read_bytes())
            except:
                pass

        self.choose_preview_history = False
        self.batch_size = self.load_or_def_option('batch_size', 1)
        #####

        io.input_skip_pending()
        self.on_initialize_options()

        if self.is_first_run():
            # 仅在第一次运行模型初始化时保存为默认选项
            self.default_options_path.write_bytes(pickle.dumps(self.options))

        self.autobackup_hour = self.options.get('autobackup_hour', 0)
        self.write_preview_history = self.options.get('write_preview_history', False)
        self.target_iter = self.options.get('target_iter', 0)
        self.random_flip = self.options.get('random_flip', True)
        self.random_src_flip = self.options.get('random_src_flip', False)
        self.random_dst_flip = self.options.get('random_dst_flip', True)

        self.on_initialize()
        self.options['batch_size'] = self.batch_size

        self.preview_history_writer = None
        if self.is_training:
            self.preview_history_path = self.saved_models_path / (f'{self.get_model_name()}_history')
            self.autobackups_path = self.saved_models_path / (f'{self.get_model_name()}_autobackups')

            if self.write_preview_history or io.is_colab():
                if not self.preview_history_path.exists():
                    self.preview_history_path.mkdir(exist_ok=True)
                else:
                    if self.iter == 0:
                        for filename in pathex.get_image_paths(self.preview_history_path):
                            Path(filename).unlink()

            if self.generator_list is None:
                raise ValueError('您未设置 set_training_data_generators()')
            else:
                for i, generator in enumerate(self.generator_list):
                    if not isinstance(generator, SampleGeneratorBase):
                        raise ValueError('训练数据生成器不是 SampleGeneratorBase 的子类')

            self.update_sample_for_preview(choose_preview_history=self.choose_preview_history)

            if self.autobackup_hour != 0:
                self.autobackup_start_time = time.time()

                if not self.autobackups_path.exists():
                    self.autobackups_path.mkdir(exist_ok=True)

        io.log_info(self.get_summary_text())



    def update_sample_for_preview(self, choose_preview_history=False, force_new=False):
        # 更新预览样本
        if self.sample_for_preview is None or choose_preview_history or force_new:
            if choose_preview_history and io.is_support_windows():
                wnd_name = "[p] - next. [space] - switch preview type. [enter] - confirm."
                io.log_info(f"选择预览历史的图像。 {wnd_name}")
                io.named_window(wnd_name)
                io.capture_keys(wnd_name)
                choosed = False
                preview_id_counter = 0
                while not choosed:
                    self.sample_for_preview = self.generate_next_samples()
                    previews = self.get_history_previews()

                    io.show_image(wnd_name, (previews[preview_id_counter % len(previews)][1] * 255).astype(np.uint8))

                    while True:
                        key_events = io.get_key_events(wnd_name)
                        key, chr_key, ctrl_pressed, alt_pressed, shift_pressed = key_events[-1] if len(key_events) > 0 else (0, 0, False, False, False)
                        if key == ord('\n') or key == ord('\r'):
                            choosed = True
                            break
                        elif key == ord(' '):
                            preview_id_counter += 1
                            break
                        elif key == ord('p'):
                            break

                        try:
                            io.process_messages(0.1)
                        except KeyboardInterrupt:
                            choosed = True

                io.destroy_window(wnd_name)
            else:
                self.sample_for_preview = self.generate_next_samples()

        try:
            self.get_history_previews()
        except:
            self.sample_for_preview = self.generate_next_samples()

        self.last_sample = self.sample_for_preview

    def load_or_def_option(self, name, def_value):
        # 加载或默认选项
        options_val = self.options.get(name, None)
        if options_val is not None:
            return options_val

        def_opt_val = self.default_options.get(name, None)
        if def_opt_val is not None:
            return def_opt_val

        return def_value

    def ask_override(self):
        # 询问是否覆盖
        return self.is_training and self.iter != 0 and io.input_in_time("在2秒内按回车键以覆盖模型设置。", 5 if io.is_colab() else 2)

    def ask_autobackup_hour(self, default_value=0):
        # 询问自动备份小时
        default_autobackup_hour = self.options['autobackup_hour'] = self.load_or_def_option('autobackup_hour', default_value)
        self.options['autobackup_hour'] = io.input_int(f"每 N 小时自动备份", default_autobackup_hour, add_info="0..24", help_message="每 N 小时自动备份模型文件和预览。最新的备份位于模型/<>_autobackups/01文件夹中。")

    def ask_write_preview_history(self, default_value=False):
        # 询问是否写入预览历史
        default_write_preview_history = self.load_or_def_option('write_preview_history', default_value)
        self.options['write_preview_history'] = io.input_bool(f"写入预览历史", default_write_preview_history, help_message="预览历史将写入到 <ModelName>_history 文件夹中。")

        if self.options['write_preview_history']:
            if io.is_support_windows():
                self.choose_preview_history = io.input_bool("选择用于预览历史的图像", False)
            elif io.is_colab():
                self.choose_preview_history = io.input_bool("随机选择新的图像以用于预览历史", False, help_message="如果您在不同的名人上重用相同的模型，则预览图像历史将保持在旧的人脸上。除非您要更改 src/dst 为新的人。")

    def ask_target_iter(self, default_value=0):
        # 询问目标迭代次数
        default_target_iter = self.load_or_def_option('target_iter', default_value)
        self.options['target_iter'] = max(0, io.input_int("目标迭代次数", default_target_iter))

    def ask_random_flip(self):
        # 询问是否随机翻转
        default_random_flip = self.load_or_def_option('random_flip', True)
        self.options['random_flip'] = io.input_bool("随机翻转面部", default_random_flip, help_message="启用此选项后，预测的面部看起来更自然，但 src 面部集必须涵盖所有面部方向作为 dst 面部集。")

    def ask_random_src_flip(self):
        # 询问是否随机翻转 SRC
        default_random_src_flip = self.load_or_def_option('random_src_flip', False)
        self.options['random_src_flip'] = io.input_bool("随机翻转 SRC 面部", default_random_src_flip, help_message="随机水平翻转 SRC 面部集。覆盖更多角度，但面部可能看起来不太自然。")

    def ask_random_dst_flip(self):
        # 询问是否随机翻转 DST
        default_random_dst_flip = self.load_or_def_option('random_dst_flip', True)
        self.options['random_dst_flip'] = io.input_bool("随机翻转 DST 面部", default_random_dst_flip, help_message="随机水平翻转 DST 面部集。如果未启用 src 随机翻转，则使 src->dst 的泛化更好。")

    def ask_batch_size(self, suggest_batch_size=None, range=None):
        # 询问批处理大小
        default_batch_size = self.load_or_def_option('batch_size', suggest_batch_size or self.batch_size)

        batch_size = max(0, io.input_int("批处理大小", default_batch_size, valid_range=range, help_message="更大的批处理大小对于神经网络的泛化效果更好，但可能会导致内存不足错误。手动调整此值以适应您的显卡。"))

        if range is not None:
            batch_size = np.clip(batch_size, range[0], range[1])

        self.options['batch_size'] = self.batch_size = batch_size

    #overridable
    def on_initialize_options(self):
        pass

    #overridable
    def on_initialize(self):
        '''
        初始化模型

        存储和检索 self.options[''] 中的模型选项

        检查示例
        '''
        pass

    #overridable
    def onSave(self):
        # 在此处保存您的模型
        pass

    # ，用于训练模型
    #overridable
    def onTrainOneIter(self, sample, generator_list):
        # 在此处训练模型

        # 返回损失的数组，例如 (('loss_src', 0), ('loss_dst', 0))
        return (('loss_src', 0), ('loss_dst', 0))

    #，用于获取预览图像
    #overridable
    def onGetPreview(self, sample, for_history=False):
        # 您可以返回多个预览图像
        # 返回格式为 [ ('preview_name', preview_rgb), ... ]
        return []

    # 如果要使模型名称与文件夹名称不同，可重写此方法
    def get_model_name(self):
        return self.model_name

    # ，返回模型文件名列表
    #overridable
    def get_model_filename_list(self):
        return []

    # ，返回MergerConfig用于模型
    #overridable
    def get_MergerConfig(self):
        # 应该返回 predictor_func、predictor_input_shape 和 MergerConfig() 用于模型
        raise NotImplementedError

    # 获取预训练数据路径
    def get_pretraining_data_path(self):
        return self.pretraining_data_path

    # 获取目标迭代次数
    def get_target_iter(self):
        return self.target_iter

    # 检查是否达到目标迭代次数
    def is_reached_iter_goal(self):
        return self.target_iter != 0 and self.iter >= self.target_iter

    # 获取当前预览图像
    def get_previews(self):
        return self.onGetPreview(self.last_sample)

    # 获取历史预览图像
    def get_history_previews(self):
        return self.onGetPreview(self.sample_for_preview, for_history=True)

    # 获取预览历史记录编写器
    def get_preview_history_writer(self):
        if self.preview_history_writer is None:
            self.preview_history_writer = PreviewHistoryWriter()
        return self.preview_history_writer

    # 保存模型和相关数据
    def save(self):
        # 将摘要文本保存到文件中
        Path(self.get_summary_path()).write_text(self.get_summary_text())

        # 调用onSave方法，可以在此方法中保存模型权重等
        self.onSave()

        # 创建模型数据字典
        model_data = {
            'iter': self.iter,
            'options': self.options,
            'loss_history': self.loss_history,
            'sample_for_preview': self.sample_for_preview,
            'choosed_gpu_indexes': self.choosed_gpu_indexes,
        }

        # 将模型数据保存到文件中
        pathex.write_bytes_safe(self.model_data_path, pickle.dumps(model_data))

        # 如果启用了自动备份
        if self.autobackup_hour != 0:
            diff_hour = int((time.time() - self.autobackup_start_time) // 3600)

            # 检查是否应该执行自动备份
            if diff_hour > 0 and diff_hour % self.autobackup_hour == 0:
                self.autobackup_start_time += self.autobackup_hour * 3600
                self.create_backup()

    # 创建模型备份
    def create_backup(self):
        io.log_info("创建备份...", end='\r')

        if not self.autobackups_path.exists():
            self.autobackups_path.mkdir(exist_ok=True)

        # 获取所有需要备份的文件名列表
        bckp_filename_list = [self.get_strpath_storage_for_file(filename) for _, filename in self.get_model_filename_list()]
        bckp_filename_list += [str(self.get_summary_path()), str(self.model_data_path)]

        for i in range(24, 0, -1):
            idx_str = '%.2d' % i
            next_idx_str = '%.2d' % (i + 1)

            idx_backup_path = self.autobackups_path / idx_str
            next_idx_packup_path = self.autobackups_path / next_idx_str

            if idx_backup_path.exists():
                if i == 24:
                    pathex.delete_all_files(idx_backup_path)
                else:
                    next_idx_packup_path.mkdir(exist_ok=True)
                    pathex.move_all_files(idx_backup_path, next_idx_packup_path)

            if i == 1:
                idx_backup_path.mkdir(exist_ok=True)
                for filename in bckp_filename_list:
                    shutil.copy(str(filename), str(idx_backup_path / Path(filename).name))

                previews = self.get_previews()
                plist = []
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [(bgr, idx_backup_path / (('preview_%s.jpg') % (name)))]

                if len(plist) != 0:
                    self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

    # 用于调试一个迭代
    def debug_one_iter(self):
        images = []
        for generator in self.generator_list:
            for i, batch in enumerate(next(generator)):
                if len(batch.shape) == 4:
                    images.append(batch[0])

        return imagelib.equalize_and_stack_square(images)

    # 生成下一个样本
    def generate_next_samples(self):
        sample = []
        for generator in self.generator_list:
            if generator.is_initialized():
                sample.append(generator.generate_next())
            else:
                sample.append([])
        self.last_sample = sample
        return sample

    # 是否应该保存预览历史记录
    #overridable
    def should_save_preview_history(self):
        # 如果不在Colab中，每迭代10次保存一次预览历史记录，如果在Colab中，每迭代100次保存一次
        return (not io.is_colab() and self.iter % 10 == 0) or (io.is_colab() and self.iter % 100 == 0)

    # 训练一个迭代
    def train_one_iter(self):
        # 记录迭代开始时间
        iter_time = time.time()
        
        # 调用onTrainOneIter方法进行模型训练，获取损失信息
        losses = self.onTrainOneIter()
        
        # 计算迭代时间
        iter_time = time.time() - iter_time
        
        # 将损失信息添加到损失历史记录中
        self.loss_history.append([float(loss[1]) for loss in losses])

        # 如果应该保存预览历史记录
        if self.should_save_preview_history():
            plist = []

            if io.is_colab():
                # 如果在Colab中，获取预览并保存
                previews = self.get_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    plist += [(bgr, self.get_strpath_storage_for_file('preview_%s.jpg' % (name)))]

            if self.write_preview_history:
                # 如果设置了写入预览历史记录，获取历史预览并保存
                previews = self.get_history_previews()
                for i in range(len(previews)):
                    name, bgr = previews[i]
                    path = self.preview_history_path / name
                    plist += [(bgr, str(path / (f'{self.iter:07d}.jpg')))]
                    if not io.is_colab():
                        plist += [(bgr, str(path / ('_last.jpg')))]

            # 如果有预览需要保存，调用预览历史记录写入器来保存
            if len(plist) != 0:
                self.get_preview_history_writer().post(plist, self.loss_history, self.iter)

        # 更新迭代计数器
        self.iter += 1

        return self.iter, iter_time

    # 跳过一个迭代
    def pass_one_iter(self):
        self.generate_next_samples()

    # 完成训练后的清理工作
    def finalize(self):
        nn.close_session()

    # 是否第一次运行
    def is_first_run(self):
        return self.iter == 0

    # 是否处于调试模式
    def is_debug(self):
        return self.debug

    # 设置批量大小
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # 获取批量大小
    def get_batch_size(self):
        return self.batch_size

    # 获取迭代次数
    def get_iter(self):
        return self.iter

    # 设置迭代次数
    def set_iter(self, iter):
        self.iter = iter
        self.loss_history = self.loss_history[:iter]

    # 获取损失历史记录
    def get_loss_history(self):
        return self.loss_history

    # 设置训练数据生成器列表
    def set_training_data_generators(self, generator_list):
        self.generator_list = generator_list

    # 获取训练数据生成器列表
    def get_training_data_generators(self):
        return self.generator_list

    # 获取模型根路径
    def get_model_root_path(self):
        return self.saved_models_path

    # 获取存储文件的字符串路径
    def get_strpath_storage_for_file(self, filename):
        return str(self.saved_models_path / (self.get_model_name() + '_' + filename))

    # 获取摘要文件的路径
    def get_summary_path(self):
        return self.get_strpath_storage_for_file('summary.txt')

    # 获取模型摘要文本
    def get_summary_text(self):
        # 复制可见选项
        visible_options = self.options.copy()
        visible_options.update(self.options_show_override)
        
        # 生成模型超参数的文本摘要
        # 找到最长的键名和值字符串，用作列宽
        width_name = max([len(k) for k in visible_options.keys()] + [17]) + 1  # 左边缘的单个空格缓冲。最小为17，最长静态字符串的长度为"Current iteration"
        width_value = max([len(str(x)) for x in visible_options.values()] + [len(str(self.get_iter())), len(self.get_model_name())]) + 1  # 右边缘的单个空格缓冲
        if len(self.device_config.devices) != 0:  # 检查GPU名称的长度
            width_value = max([len(device.name) + 1 for device in self.device_config.devices] + [width_value])
        width_total = width_name + width_value + 2  # 加2表示": "的长度

        summary_text = []
        summary_text += [f'=={" Model Summary ":=^{width_total}}==']  # 模型/状态摘要
        summary_text += [f'=={" " * width_total}==']
        summary_text += [f'=={"Model name": >{width_name}}: {self.get_model_name(): <{width_value}}==']  # 名称
        summary_text += [f'=={" " * width_total}==']
        summary_text += [f'=={"Current iteration": >{width_name}}: {str(self.get_iter()): <{width_value}}==']  # 迭代次数
        summary_text += [f'=={" " * width_total}==']

        summary_text += [f'=={" Model Options ":-^{width_total}}==']  # 模型选项
        summary_text += [f'=={" " * width_total}==']
        for key in visible_options.keys():
            summary_text += [f'=={key: >{width_name}}: {str(visible_options[key]): <{width_value}}==']  # 可见选项的键/值对
        summary_text += [f'=={" " * width_total}==']

        summary_text += [f'=={" Running On ":-^{width_total}}==']  # 训练硬件信息
        summary_text += [f'=={" " * width_total}==']
        if len(self.device_config.devices) == 0:
            summary_text += [f'=={"Using device": >{width_name}}: {"CPU": <{width_value}}==']  # 仅CPU
        else:
            for device in self.device_config.devices:
                summary_text += [f'=={"Device index": >{width_name}}: {device.index: <{width_value}}==']  # GPU硬件设备索引
                summary_text += [f'=={"Name": >{width_name}}: {device.name: <{width_value}}==']  # GPU名称
                vram_str = f'{device.total_mem_gb:.2f}GB'  # GPU VRAM - 格式为#.##（或##.##）
                summary_text += [f'=={"VRAM": >{width_name}}: {vram_str: <{width_value}}==']
        summary_text += [f'=={" " * width_total}==']
        summary_text += [f'=={"=" * width_total}==']
        summary_text = "\n".join(summary_text)
        return summary_text

    # 获取损失历史记录的预览图像
    @staticmethod
    def get_loss_history_preview(loss_history, iter, w, c):
        # 复制损失历史记录
        loss_history = np.array(loss_history.copy())

        lh_height = 100
        lh_img = np.ones((lh_height, w, c)) * 0.1

        if len(loss_history) != 0:
            loss_count = len(loss_history[0])
            lh_len = len(loss_history)

            l_per_col = lh_len / w
            plist_max = [[max(0.0, loss_history[int(col * l_per_col)][p],
                            *[loss_history[i_ab][p]
                                for i_ab in range(int(col * l_per_col), int((col + 1) * l_per_col))
                                ]
                            )
                        for p in range(loss_count)
                        ]
                        for col in range(w)
                        ]

            plist_min = [[min(plist_max[col][p], loss_history[int(col * l_per_col)][p],
                            *[loss_history[i_ab][p]
                                for i_ab in range(int(col * l_per_col), int((col + 1) * l_per_col))
                                ]
                            )
                        for p in range(loss_count)
                        ]
                        for col in range(w)
                        ]

            plist_abs_max = np.mean(loss_history[len(loss_history) // 5:]) * 2

            for col in range(0, w):
                for p in range(0, loss_count):
                    point_color = [1.0] * c
                    point_color[0:3] = colorsys.hsv_to_rgb(p * (1.0 / loss_count), 1.0, 1.0)

                    ph_max = int((plist_max[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_max = np.clip(ph_max, 0, lh_height - 1)

                    ph_min = int((plist_min[col][p] / plist_abs_max) * (lh_height - 1))
                    ph_min = np.clip(ph_min, 0, lh_height - 1)

                    for ph in range(ph_min, ph_max + 1):
                        lh_img[(lh_height - ph - 1), col] = point_color

        lh_lines = 5
        lh_line_height = (lh_height - 1) / lh_lines
        for i in range(0, lh_lines + 1):
            lh_img[int(i * lh_line_height), :] = (0.8,) * c

        last_line_t = int((lh_lines - 1) * lh_line_height)
        last_line_b = int(lh_lines * lh_line_height)

        lh_text = 'Iter: %d' % (iter) if iter != 0 else ''

        lh_img[last_line_t:last_line_b, 0:w] += imagelib.get_text_image((last_line_b - last_line_t, w, c), lh_text,
                                                                    color=[0.8] * c)
        return lh_img


class PreviewHistoryWriter():
    def __init__(self):
        # 创建一个多进程队列
        self.sq = multiprocessing.Queue()
        # 创建一个多进程子进程来处理队列中的数据
        self.p = multiprocessing.Process(target=self.process, args=(self.sq,))
        self.p.daemon = True
        self.p.start()

    # 子进程函数，用于处理队列中的数据
    def process(self, sq):
        while True:
            while not sq.empty():
                # 从队列中获取预览图像列表、损失历史和迭代次数
                plist, loss_history, iter = sq.get()

                # 预览图像和损失历史的缓存
                preview_lh_cache = {}

                for preview, filepath in plist:
                    filepath = Path(filepath)
                    i = (preview.shape[1], preview.shape[2])

                    # 获取缓存中的预览图像和损失历史，如果没有则生成
                    preview_lh = preview_lh_cache.get(i, None)
                    if preview_lh is None:
                        preview_lh = ModelBase.get_loss_history_preview(loss_history, iter, preview.shape[1], preview.shape[2])
                        preview_lh_cache[i] = preview_lh

                    # 将预览图像和损失历史合并，并转换为图像格式
                    img = (np.concatenate([preview_lh, preview], axis=0) * 255).astype(np.uint8)

                    # 创建目录并保存图像文件
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(filepath, img)

            time.sleep(0.01)

    # 将数据添加到队列中以进行处理
    def post(self, plist, loss_history, iter):
        self.sq.put((plist, loss_history, iter))

    # 禁用对象的序列化
    def __getstate__(self):
        return dict()

    def __setstate__(self, d):
        self.__dict__.update(d)
