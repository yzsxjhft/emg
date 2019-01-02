from time import sleep
from myo import init, Hub, DeviceListener


class Listener(DeviceListener):

    def on_pair(self, myo, timestamp, firmware_version):
        print("Hello, Myo!")

    def on_unpair(self, myo, timestamp):
        print("Goodbye, Myo!")

    def on_pose(self, timestamp, pose):
        print(pose)

    def on_emg_data(self, timestamp, emg):
        print(emg)


init(r"D:\desktop\myo\myo-svm\myo-sdk-win-0.9.0")
listener = DeviceListener()
hub = Hub()
hub.run(1000, listener)

try:
    while True:
        sleep(0.5)
finally:
    hub.shutdown()
