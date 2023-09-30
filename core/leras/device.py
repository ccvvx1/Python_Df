import sys
import ctypes
import os
import multiprocessing
import json
import time
from pathlib import Path
from core.interact import interact as io

# 导入必要的库和模块
# 这段代码定义了两个类：Device 和 Devices。
# Device 类表示一个具体的计算设备，它包含设备的索引、类型、名称、总内存和可用内存等信息。
# Devices 类表示一组计算设备，它包含多个 Device 对象。该类提供了一些方法，如获取设备数量、按索引访问设备、获取可用内存最多的设备以及获取可用内存最少的设备。
# 请注意，这段代码的注释已经包含了详细的解释。此外，代码中还导入了一些外部库和模块，但这些库和模块的具体用途在代码中并没有详细说明。如果需要更多关于这些库和模块的信息，你可能需要查看导入语句所在的上下文来了解它们的用途。
class Device(object):
    def __init__(self, index, tf_dev_type, name, total_mem, free_mem):
        # 初始化设备对象
        self.index = index  # 设备索引
        self.tf_dev_type = tf_dev_type  # 设备类型
        self.name = name  # 设备名称
        self.total_mem = total_mem  # 总内存大小（以字节为单位）
        self.total_mem_gb = total_mem / 1024**3  # 总内存大小（以GB为单位）
        self.free_mem = free_mem  # 可用内存大小（以字节为单位）
        self.free_mem_gb = free_mem / 1024**3  # 可用内存大小（以GB为单位）

    def __str__(self):
        # 用于在打印时显示设备信息的字符串表示
        return f"[{self.index}]:[{self.name}][{self.free_mem_gb:.3}/{self.total_mem_gb:.3}]"

class Devices(object):
    all_devices = None

    def __init__(self, devices):
        # 初始化设备集合对象
        self.devices = devices

    def __len__(self):
        # 返回设备集合中设备的数量
        return len(self.devices)

    def __getitem__(self, key):
        # 获取指定索引位置的设备
        result = self.devices[key]
        if isinstance(key, slice):
            return Devices(result)
        return result

    def __iter__(self):
        # 遍历设备集合
        for device in self.devices:
            yield device

    def get_best_device(self):
        # 获取可用内存最多的设备
        result = None
        idx_mem = 0
        for device in self.devices:
            mem = device.total_mem
            if mem > idx_mem:
                result = device
                idx_mem = mem
        return result

    def get_worst_device(self):
        # 获取可用内存最少的设备
        result = None
        idx_mem = sys.maxsize
        for device in self.devices:
            mem = device.total_mem
            if mem < idx_mem:
                result = device
                idx_mem = mem
        return result
    def get_device_by_index(self, idx):
        # 通过设备索引查找并返回设备对象
        for device in self.devices:
            if device.index == idx:
                return device
        return None

    def get_devices_from_index_list(self, idx_list):
        # 根据索引列表获取一组设备对象并返回
        result = []
        for device in self.devices:
            if device.index in idx_list:
                result += [device]
        return Devices(result)

    def get_equal_devices(self, device):
        # 获取与给定设备具有相同名称的设备列表
        device_name = device.name
        result = []
        for device in self.devices:
            if device.name == device_name:
                result.append(device)
        return Devices(result)

    def get_devices_at_least_mem(self, totalmemsize_gb):
        # 获取至少具有指定内存大小的设备列表（以GB为单位）
        result = []
        for device in self.devices:
            if device.total_mem >= totalmemsize_gb * (1024**3):
                result.append(device)
        return Devices(result)

    @staticmethod
    def _get_tf_devices_proc(q: multiprocessing.Queue):
        # 静态方法：用于获取TensorFlow中的计算设备信息

        # 在Windows上设置GPU缓存路径（如果适用）
        if sys.platform[0:3] == 'win':
            compute_cache_path = Path(os.environ['APPDATA']) / 'NVIDIA' / ('ComputeCache_ALL')
            os.environ['CUDA_CACHE_PATH'] = str(compute_cache_path)
            if not compute_cache_path.exists():
                io.log_info("Caching GPU kernels...")
                compute_cache_path.mkdir(parents=True, exist_ok=True)

        import tensorflow

        tf_version = tensorflow.version.VERSION
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

        from tensorflow.python.client import device_lib

        devices = []

        physical_devices = device_lib.list_local_devices()
        physical_devices_f = {}
        for dev in physical_devices:
            dev_type = dev.device_type
            dev_tf_name = dev.name
            dev_tf_name = dev_tf_name[dev_tf_name.index(dev_type):]

            dev_idx = int(dev_tf_name.split(':')[-1])

            if dev_type in ['GPU', 'DML']:
                dev_name = dev_tf_name

                dev_desc = dev.physical_device_desc
                if len(dev_desc) != 0:
                    if dev_desc[0] == '{':
                        dev_desc_json = json.loads(dev_desc)
                        dev_desc_json_name = dev_desc_json.get('name', None)
                        if dev_desc_json_name is not None:
                            dev_name = dev_desc_json_name
                    else:
                        for param, value in (v.split(':') for v in dev_desc.split(',')):
                            param = param.strip()
                            value = value.strip()
                            if param == 'name':
                                dev_name = value
                                break

                physical_devices_f[dev_idx] = (dev_type, dev_name, dev.memory_limit)

        q.put(physical_devices_f)
        time.sleep(0.1)

# 这些方法的作用如下：
# get_device_by_index(idx)：通过设备索引查找并返回设备对象。如果找不到匹配的设备，返回 None。
# get_devices_from_index_list(idx_list)：根据索引列表获取一组设备对象并返回。可以传入一个设备索引的列表，返回匹配索引的设备列表。
# get_equal_devices(device)：获取与给定设备具有相同名称的设备列表。这可以用于查找同一类型的多个设备。
# get_devices_at_least_mem(totalmemsize_gb)：获取至少具有指定内存大小（以GB为单位）的设备列表。
# _get_tf_devices_proc(q: multiprocessing.Queue)：这是一个静态方法，用于获取TensorFlow中的计算设备信息。它在Windows上设置GPU缓存路径，并使用TensorFlow库来获取计算设备的信息。获取的信息包括设备类型、名称和内存限制，并将其放入一个队列中以供其他部分的代码使用。
# 请注意，这些方法的注释提供了详细的解释，以帮助你理解它们的功能和用途。
        
    @staticmethod
    def initialize_main_env():
        # 初始化主计算环境的静态方法
        if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 0:
            return
            
        if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
            os.environ.pop('CUDA_VISIBLE_DEVICES')
        
        os.environ['TF_DIRECTML_KERNEL_CACHE_SIZE'] = '2500'
        os.environ['CUDA_​CACHE_​MAXSIZE'] = '2147483647'
        os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 仅记录TensorFlow错误信息
        
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=Devices._get_tf_devices_proc, args=(q,), daemon=True)
        p.start()
        p.join()
        
        visible_devices = q.get()

        os.environ['NN_DEVICES_INITIALIZED'] = '1'
        os.environ['NN_DEVICES_COUNT'] = str(len(visible_devices))
        
        for i in visible_devices:
            dev_type, name, total_mem = visible_devices[i]

            os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'] = dev_type
            os.environ[f'NN_DEVICE_{i}_NAME'] = name
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(total_mem)
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(total_mem)

# initialize_main_env()：这是一个静态方法，用于初始化主计算环境。它首先检查环境变量 NN_DEVICES_INITIALIZED 是否为0（尚未初始化）。然后，它设置一系列环境变量，包括TensorFlow和CUDA的相关设置，并启动一个多进程来获取计算设备信息（通过调用 _get_tf_devices_proc 方法）。获取的设备信息包括设备类型、名称和总内存，并将其存储在环境变量中。

    @staticmethod
    def getDevices():
        # 获取设备信息的静态方法
        if Devices.all_devices is None:
            if int(os.environ.get("NN_DEVICES_INITIALIZED", 0)) != 1:
                raise Exception("nn devices are not initialized. Run initialize_main_env() in the main process.")
            devices = []
            for i in range(int(os.environ['NN_DEVICES_COUNT'])):
                devices.append(Device(index=i,
                                    tf_dev_type=os.environ[f'NN_DEVICE_{i}_TF_DEV_TYPE'],
                                    name=os.environ[f'NN_DEVICE_{i}_NAME'],
                                    total_mem=int(os.environ[f'NN_DEVICE_{i}_TOTAL_MEM']),
                                    free_mem=int(os.environ[f'NN_DEVICE_{i}_FREE_MEM']), )
                                )
            Devices.all_devices = Devices(devices)

        return Devices.all_devices

# getDevices()：这是一个静态方法，用于获取设备信息。它首先检查是否已经初始化计算环境（通过检查 NN_DEVICES_INITIALIZED 环境变量），如果没有初始化，则引发异常。然后，它从环境变量中读取设备信息，创建对应的 Device 对象，并将它们存储在 Devices.all_devices 中。如果已经初始化，则直接返回存储的设备信息。
# 这两个方法的注释提供了详细的解释，以帮助你理解它们的功能和用途。它们用于管理计算设备信息和计算环境的初始化。

"""

        
        # {'name'      : name.split(b'\0', 1)[0].decode(),
        #     'total_mem' : totalMem.value
        # }

        
        
        
        
        return

        
        
        
        min_cc = int(os.environ.get("TF_MIN_REQ_CAP", 35))
        libnames = ('libcuda.so', 'libcuda.dylib', 'nvcuda.dll')
        for libname in libnames:
            try:
                cuda = ctypes.CDLL(libname)
            except:
                continue
            else:
                break
        else:
            return Devices([])

        nGpus = ctypes.c_int()
        name = b' ' * 200
        cc_major = ctypes.c_int()
        cc_minor = ctypes.c_int()
        freeMem = ctypes.c_size_t()
        totalMem = ctypes.c_size_t()

        result = ctypes.c_int()
        device = ctypes.c_int()
        context = ctypes.c_void_p()
        error_str = ctypes.c_char_p()

        devices = []

        if cuda.cuInit(0) == 0 and \
            cuda.cuDeviceGetCount(ctypes.byref(nGpus)) == 0:
            for i in range(nGpus.value):
                if cuda.cuDeviceGet(ctypes.byref(device), i) != 0 or \
                    cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) != 0 or \
                    cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) != 0:
                    continue

                if cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device) == 0:
                    if cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem)) == 0:
                        cc = cc_major.value * 10 + cc_minor.value
                        if cc >= min_cc:
                            devices.append ( {'name'      : name.split(b'\0', 1)[0].decode(),
                                              'total_mem' : totalMem.value,
                                              'free_mem'  : freeMem.value,
                                              'cc'        : cc
                                              })
                    cuda.cuCtxDetach(context)

        os.environ['NN_DEVICES_COUNT'] = str(len(devices))
        for i, device in enumerate(devices):
            os.environ[f'NN_DEVICE_{i}_NAME'] = device['name']
            os.environ[f'NN_DEVICE_{i}_TOTAL_MEM'] = str(device['total_mem'])
            os.environ[f'NN_DEVICE_{i}_FREE_MEM'] = str(device['free_mem'])
            os.environ[f'NN_DEVICE_{i}_CC'] = str(device['cc'])
"""