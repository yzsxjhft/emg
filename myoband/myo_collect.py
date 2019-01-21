from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np


class EmgCollector(myo.DeviceListener):

    def __init__(self):
        self.emg_list = [[] for x in range(8)]

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        emg_list = event.emg
        for i in range(len(emg_list)):
            length = len(self.emg_list[i])
            # if emg_list[i] > 0:
            if length >= 10:
                emg_list[i] = pow(emg_list[i], 2)
                emg_list[i] = emg_list[i] * 0.6 + 0.4 * sum(self.emg_list[i][length - 10:length - 1]) / 10
                emg_list[i] = round(emg_list[i], 2)
            self.emg_list[i].append(emg_list[i])

    def save_data(self, filename):
        f = open(filename, 'w+')
        data = zip(*self.emg_list)
        for d in data:
            f.write(str(d))
            f.write("\n")
        f.close()


if __name__ == '__main__':
    myo.init(sdk_path=r"D:\desktop\myo\myo-svm-3\myo-sdk-win-0.9.0")
    hub = myo.Hub()
    listener = EmgCollector()
    pose_data = ["left", "right", "rest", "open"]
    print("收集训练数据")
    for pose in pose_data:
        input("Action %s,按回车开始" % pose)
        hub.run(listener.on_event, 10000)
        listener.save_data(pose)