from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np


class EmgCollector(myo.DeviceListener):

    def __init__(self, n):
        self.emg_list = list()

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        print(event.emg)
        self.emg_list.append(event.emg)

    def save_data(self, filename):
        f = open(filename, 'w+')
        for d in self.emg_list:
            f.write(str(d))
            f.write("\n")
        f.close()


if __name__ == '__main__':
    myo.init(sdk_path=r"D:\desktop\myo\myo-svm-3\myo-sdk-win-0.9.0")
    hub = myo.Hub()
    listener = EmgCollector(512)
    pose_data = ["left"]
    print("收集训练数据")
    for pose in pose_data:
        input("Action %s,按回车开始" % pose)
        hub.run(listener.on_event, 10000)
        listener.save_data(pose + "1")