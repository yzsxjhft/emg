from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

import myo
import numpy as np


class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)
    self.emg_list = [[] for x in range(8)]

  def get_emg_data(self):
      with self.lock:
        return self.emg_list

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    # emg_list = [pow(abs(x), 1.5) for x in event.emg]
    emg_list = event.emg
    for i in range(len(emg_list)):
        length = len(self.emg_list[i])
        if emg_list[i] > 0:
            if length >= 5:
                emg_list[i] = pow(emg_list[i], 1.5)
                emg_list[i] = (emg_list[i]*0.6 + 0.4*sum(self.emg_list[i][length-2:length-1]))
            self.emg_list[i].append(emg_list[i])
        length = len(self.emg_list[i])
        if length >= self.n:
            self.emg_list[i].pop(0)


class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    self.fig = plt.figure()
    self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    [(ax.set_ylim([-10, 1000])) for ax in self.axes]
    self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    # emg_data = np.array([x[1] for x in emg_data]).T
    union = zip(self.graphs, emg_data)
    for g, data in union:
      if len(data) < self.n:
        # Fill the left side with zeroes.
        data = np.concatenate([np.zeros(self.n - len(data)), data])
      g.set_ydata(data)
    plt.draw()

  def main(self):
    while True:
      self.update_plot()
      plt.pause(1.0 / 30)


def main():
  myo.init(sdk_path=r"D:\desktop\myo\myo-svm-3\myo-sdk-win-0.9.0")
  hub = myo.Hub()
  listener = EmgCollector(512)
  with hub.run_in_background(listener.on_event):
    Plot(listener).main()


if __name__ == '__main__':
  main()