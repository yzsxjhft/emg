import myo
import os


class EmgCollector(myo.DeviceListener):

    def __init__(self):
        self.emg_list = []

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        emg_list = event.emg
        self.emg_list.append(emg_list)

    def save_data(self, pose, num):
        parent_dir = "data/%s" % pose
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)

        filename = "%s/%s" % (parent_dir, num)
        f = open(filename, 'w')
        data = self.emg_list
        for d in data:
            f.write(str(d))
            f.write("\n")
        f.close()
        self.emg_list = []


if __name__ == '__main__':
    myo.init(sdk_path=r"C:\Users\86188\Desktop\emg\myo-sdk-win-0.9.0")
    hub = myo.Hub()
    listener = EmgCollector()
    pose_data = ["left", "right", "rest", "open"]
    print("收集训练数据")
    for pose in pose_data:
        for i in range(1, 31):
            input("Action %s-%s,按回车开始" % (pose, i))
            hub.run(listener.on_event, 1000)
            listener.save_data(pose, i)
