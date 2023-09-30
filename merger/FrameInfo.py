from pathlib import Path

class FrameInfo(object):
    def __init__(self, filepath=None, landmarks_list=None):
        # 初始化帧信息对象
        # filepath: 帧对应的文件路径
        # landmarks_list: 帧中的关键点坐标列表，默认为空列表
        self.filepath = filepath  # 文件路径
        self.landmarks_list = landmarks_list or []  # 关键点坐标列表，如果未提供，则为空列表
        self.motion_deg = 0  # 运动角度，默认为0
        self.motion_power = 0  # 运动强度，默认为0
