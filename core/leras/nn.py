"""
Leras.

like lighter keras.
This is my lightweight neural network library written from scratch
based on pure tensorflow without keras.

Provides:
+ full freedom of tensorflow operations without keras model's restrictions
+ easy model operations like in PyTorch, but in graph mode (no eager execution)
+ convenient and understandable logic

Reasons why we cannot import tensorflow or any tensorflow.sub modules right here:
1) program is changing env variables based on DeviceConfig before import tensorflow
2) multiprocesses will import tensorflow every spawn

NCHW speed up training for 10-20%.
"""

import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
import numpy as np
from core.interact import interact as io
from .device import Devices

# 导入必要的库和模块

class nn():
    current_DeviceConfig = None

    tf = None
    tf_sess = None
    tf_sess_config = None
    tf_default_device_name = None
    
    data_format = None
    conv2d_ch_axis = None
    conv2d_spatial_axes = None

    floatx = None
    
    @staticmethod
    def initialize(device_config=None, floatx="float32", data_format="NHWC"):
        # 初始化深度学习框架和配置参数的静态方法

        if nn.tf is None:
            if device_config is None:
                device_config = nn.getCurrentDeviceConfig()
            nn.setCurrentDeviceConfig(device_config)

            # 在导入TensorFlow之前操作环境变量

            first_run = False
            if len(device_config.devices) != 0:
                if sys.platform[0:3] == 'win':
                    # Windows特定的环境变量
                    if all([x.name == device_config.devices[0].name for x in device_config.devices]):
                        devices_str = "_" + device_config.devices[0].name.replace(' ', '_')
                    else:
                        devices_str = ""
                        for device in device_config.devices:
                            devices_str += "_" + device.name.replace(' ', '_')

                    compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache' + devices_str)
                    if not compute_cache_path.exists():
                        first_run = True
                        compute_cache_path.mkdir(parents=True, exist_ok=True)
                    os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            
            if first_run:
                io.log_info("Caching GPU kernels...")

            import tensorflow

            tf_version = tensorflow.version.VERSION
            #if tf_version is None:
            #    tf_version = tensorflow.version.GIT_VERSION
            if tf_version[0] == 'v':
                tf_version = tf_version[1:]
            if tf_version[0] == '2':
                tf = tensorflow.compat.v1
            else:
                tf = tensorflow

            import logging
            # 禁用TensorFlow的警告信息
            tf_logger = logging.getLogger('tensorflow')
            tf_logger.setLevel(logging.ERROR)
            
            if tf_version[0] == '2':
                tf.disable_v2_behavior()
            nn.tf = tf

            # 初始化深度学习框架
            import core.leras.ops
            import core.leras.layers
            import core.leras.initializers
            import core.leras.optimizers
            import core.leras.models
            import core.leras.archis
            
            # 配置TensorFlow会话选项
            if len(device_config.devices) == 0:
                config = tf.ConfigProto(device_count={'GPU': 0})
                nn.tf_default_device_name = '/CPU:0'
            else:
                nn.tf_default_device_name = f'/{device_config.devices[0].tf_dev_type}:0'
                
                config = tf.ConfigProto()
                config.gpu_options.visible_device_list = ','.join([str(device.index) for device in device_config.devices])
                
            config.gpu_options.force_gpu_compatible = True
            config.gpu_options.allow_growth = True
            nn.tf_sess_config = config
            
        if nn.tf_sess is None:
            nn.tf_sess = tf.Session(config=nn.tf_sess_config)

        if floatx == "float32":
            floatx = nn.tf.float32
        elif floatx == "float16":
            floatx = nn.tf.float16
        else:
            raise ValueError(f"unsupported floatx {floatx}")
        nn.set_floatx(floatx)
        nn.set_data_format(data_format)  

        # 这段代码定义了一个名为 nn 的类，用于初始化深度学习框架（例如TensorFlow）以及配置相关参数。
        # initialize(device_config=None, floatx="float32", data_format="NHWC")：这是一个静态方法，用于初始化深度学习框架和配置参数。它接受三个可选参数：
        # device_config：设备配置对象，用于配置计算设备。
        # floatx：指定浮点数数据类型（默认为 "float32"）。
        # data_format：指定数据格式（默认为 "NHWC"）。
        # 在方法内部，首先检查是否已经初始化TensorFlow和相关参数，如果没有初始化，则进行初始化。
        # 然后，根据计算设备配置设置相关的环境变量，包括GPU缓存路径等。
        # 接着，导入TensorFlow库，并设置一些TensorFlow的配置选项，以便在需要时使用GPU。
        # 最后，根据指定的浮点数数据类型和数据格式进行配置，并初始化深度学习框架。
        # 这段代码的注释提供了详细的解释，以帮助你理解它的功能和用途。它用于配置深度学习框架和相关参数，以便进行深度学习任务。

    @staticmethod
    def initialize_main_env():
        # 初始化主计算环境的静态方法
        Devices.initialize_main_env()

    @staticmethod
    def set_floatx(tf_dtype):
        """
        设置默认的浮点数数据类型，适用于所有层当它们的数据类型为 None 时。
        """
        nn.floatx = tf_dtype

    @staticmethod
    def set_data_format(data_format):
        """
        设置数据格式（data_format）。

        参数：
        - data_format：数据格式字符串，支持 "NHWC" 或 "NCHW"。
        """
        if data_format != "NHWC" and data_format != "NCHW":
            raise ValueError(f"不支持的数据格式 {data_format}")
        nn.data_format = data_format

        if data_format == "NHWC":
            nn.conv2d_ch_axis = 3
            nn.conv2d_spatial_axes = [1, 2]
        elif data_format == "NCHW":
            nn.conv2d_ch_axis = 1
            nn.conv2d_spatial_axes = [2, 3]

    @staticmethod
    def get4Dshape(w, h, c):
        """
        根据当前的数据格式返回 4D 形状。

        参数：
        - w：宽度
        - h：高度
        - c：通道数

        返回值：返回一个形状为 (None, h, w, c) 或 (None, c, h, w) 的元组，具体取决于当前的数据格式。
        """
        if nn.data_format == "NHWC":
            return (None, h, w, c)
        else:
            return (None, c, h, w)

    @staticmethod
    def to_data_format(x, to_data_format, from_data_format):
        """
        将数据从一个数据格式转换为另一个数据格式。

        参数：
        - x：要转换的数据
        - to_data_format：目标数据格式字符串，支持 "NHWC" 或 "NCHW"。
        - from_data_format：原始数据格式字符串，支持 "NHWC" 或 "NCHW"。

        返回值：返回转换后的数据。
        """
        if to_data_format == from_data_format:
            return x

        if to_data_format == "NHWC":
            return np.transpose(x, (0, 2, 3, 1))
        elif to_data_format == "NCHW":
            return np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError(f"不支持的目标数据格式 {to_data_format}")

    @staticmethod
    def getCurrentDeviceConfig():
        """
        获取当前设备配置对象。

        返回值：返回当前设备配置对象。
        """
        if nn.current_DeviceConfig is None:
            nn.current_DeviceConfig = DeviceConfig.BestGPU()  # 假设使用最佳GPU设备配置
        return nn.current_DeviceConfig

    # 这些静态方法提供了一些配置和工具函数，用于设置默认的浮点数数据类型、数据格式，进行数据格式转换，获取设备配置等。它们用于方便地配置和管理深度学习框架的参数和环境。注释提供了详细的解释，以帮助你理解它们的功能和用途。

    @staticmethod
    def setCurrentDeviceConfig(device_config):
        """
        设置当前的设备配置对象。

        参数：
        - device_config：设备配置对象，用于指定计算设备。

        用法示例：
        nn.setCurrentDeviceConfig(device_config)
        """
        nn.current_DeviceConfig = device_config

    @staticmethod
    def reset_session():
        """
        重置TensorFlow会话。

        用法示例：
        nn.reset_session()
        """
        if nn.tf is not None:
            if nn.tf_sess is not None:
                nn.tf.reset_default_graph()
                nn.tf_sess.close()
                nn.tf_sess = nn.tf.Session(config=nn.tf_sess_config)

    @staticmethod
    def close_session():
        """
        关闭TensorFlow会话。

        用法示例：
        nn.close_session()
        """
        if nn.tf_sess is not None:
            nn.tf.reset_default_graph()
            nn.tf_sess.close()
            nn.tf_sess = None

    @staticmethod
    def ask_choose_device_idxs(choose_only_one=False, allow_cpu=True, suggest_best_multi_gpu=False, suggest_all_gpu=False):
        """
        提示用户选择计算设备的索引。

        参数：
        - choose_only_one：是否只能选择一个设备（默认为 False）。
        - allow_cpu：是否允许选择 CPU 设备（默认为 True）。
        - suggest_best_multi_gpu：是否建议选择最佳的多GPU配置（默认为 False）。
        - suggest_all_gpu：是否建议选择所有GPU设备（默认为 False）。

        返回值：返回用户选择的设备索引列表。

        用法示例：
        indexes = nn.ask_choose_device_idxs(choose_only_one=True)
        """
        devices = Devices.getDevices()
        if len(devices) == 0:
            return []

        all_devices_indexes = [device.index for device in devices]

        if choose_only_one:
            suggest_best_multi_gpu = False
            suggest_all_gpu = False

        if suggest_all_gpu:
            best_device_indexes = all_devices_indexes
        elif suggest_best_multi_gpu:
            best_device_indexes = [device.index for device in devices.get_equal_devices(devices.get_best_device())]
        else:
            best_device_indexes = [devices.get_best_device().index]
        best_device_indexes = ",".join([str(x) for x in best_device_indexes])

        io.log_info("")
        if choose_only_one:
            io.log_info("选择一个GPU索引。")
        else:
            io.log_info("选择一个或多个GPU索引（用逗号分隔）。")
        io.log_info("")

        if allow_cpu:
            io.log_info("[CPU] : CPU")
        for device in devices:
            io.log_info(f"  [{device.index}] : {device.name}")

        io.log_info("")

        while True:
            try:
                if choose_only_one:
                    choosed_idxs = io.input_str("选择哪个GPU索引?", best_device_indexes)
                else:
                    choosed_idxs = io.input_str("选择哪些GPU索引?", best_device_indexes)

                if allow_cpu and choosed_idxs.lower() == "cpu":
                    choosed_idxs = []
                    break

                choosed_idxs = [int(x) for x in choosed_idxs.split(',')]

                if choose_only_one:
                    if len(choosed_idxs) == 1:
                        break
                else:
                    if all([idx in all_devices_indexes for idx in choosed_idxs]):
                        break
            except:
                pass
        io.log_info("")

        return choosed_idxs


    # 这些静态方法提供了一些功能，包括设置当前设备配置、重置和关闭TensorFlow会话以及提示用户选择计算设备的索引。注释提供了详细的解释，以帮助你理解它们的功能和用途。这些方法用于配置和管理深度学习框架的计算设备和会话。

    class DeviceConfig():
        @staticmethod
        def ask_choose_device(*args, **kwargs):
            """
            提示用户选择计算设备，并返回一个包含所选设备索引的 DeviceConfig 对象。

            参数：与 nn.ask_choose_device_idxs 方法相同的参数。

            返回值：一个包含所选设备索引的 DeviceConfig 对象。

            用法示例：
            device_config = DeviceConfig.ask_choose_device(choose_only_one=True)
            """
            return nn.DeviceConfig.GPUIndexes(nn.ask_choose_device_idxs(*args, **kwargs))

        def __init__(self, devices=None):
            """
            初始化 DeviceConfig 对象。

            参数：
            - devices：设备列表或 Devices 对象（默认为 None）。

            用法示例：
            device_config = DeviceConfig(devices=my_devices)
            """
            devices = devices or []

            if not isinstance(devices, Devices):
                devices = Devices(devices)

            self.devices = devices
            self.cpu_only = len(devices) == 0

        @staticmethod
        def BestGPU():
            """
            创建一个 DeviceConfig 对象，包含最佳GPU设备。

            返回值：一个包含最佳GPU设备的 DeviceConfig 对象。

            用法示例：
            best_config = DeviceConfig.BestGPU()
            """
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_best_device()])

        @staticmethod
        def WorstGPU():
            """
            创建一个 DeviceConfig 对象，包含最差GPU设备。

            返回值：一个包含最差GPU设备的 DeviceConfig 对象。

            用法示例：
            worst_config = DeviceConfig.WorstGPU()
            """
            devices = Devices.getDevices()
            if len(devices) == 0:
                return nn.DeviceConfig.CPU()

            return nn.DeviceConfig([devices.get_worst_device()])

        @staticmethod
        def GPUIndexes(indexes):
            """
            创建一个 DeviceConfig 对象，包含指定索引的GPU设备。

            参数：
            - indexes：要包含的GPU设备索引列表。

            返回值：一个包含指定GPU设备索引的 DeviceConfig 对象。

            用法示例：
            gpu_config = DeviceConfig.GPUIndexes([0, 1, 2])
            """
            if len(indexes) != 0:
                devices = Devices.getDevices().get_devices_from_index_list(indexes)
            else:
                devices = []

            return nn.DeviceConfig(devices)

        @staticmethod
        def CPU():
            """
            创建一个 DeviceConfig 对象，表示仅使用CPU设备。

            返回值：一个表示仅使用CPU设备的 DeviceConfig 对象。

            用法示例：
            cpu_config = DeviceConfig.CPU()
            """
            return nn.DeviceConfig([])

    # DeviceConfig 类用于配置计算设备的对象。它包含了一些静态方法，用于创建不同的设备配置。这些方法允许用户选择最佳或最差的GPU设备，或者自定义选择特定的GPU设备。用户还可以创建仅使用CPU设备的配置。这些方法提供了灵活的选项，以满足不同深度学习任务的需求。注释提供了详细的解释，以帮助你理解它们的功能和用途。