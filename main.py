import serial
import numpy
from sklearn.naive_bayes import GaussianNB


class EmgModel:
    def __init__(self, labels):
        self.model = GaussianNB()
        self.serial = serial.Serial('COM5', 9600)
        self.labels = labels

        self.train_data = []
        self.train_target = []
        self.train_row_data = []

        self.predict_data = []
        self.predict_target = []
        self.predict_row_data = []

        self.train_num = 3
        self.predict_num = 3

    def collect_signal(self, serial_data, mean_data):
        try:
            value = str(self.serial.readline())
            value = value.replace("b", "").replace("\\r\\n", "").replace("\'", "")
            value = float(value.split(",")[1])
            print(value)
        except IndexError:
            print("Index Error")
            return
        serial_data.append(value)
        if len(serial_data) >= 5:
            mean_data.append(sum(serial_data[len(serial_data) - 5::1]) / 5)
        else:
            mean_data.append(value)

    def validate_train_data(self, pose, data):
        wave_data = self.get_wave(data)
        if len(wave_data) > 0:
            self.train_row_data.append(wave_data)
            self.train_target.append(pose)
            return True
        else:
            return False

    def validate_predict_data(self, pose, data):
        wave_data = self.get_wave(data)
        if len(wave_data) > 0:
            self.predict_row_data.append(wave_data)
            self.predict_target.append(pose)
            return True
        else:
            return False

    def get_wave(self, data):
        theta = 70
        start = 0
        end = 0
        result = []
        for i in range(len(data) - 1):
            for j in range(i, len(data) - 1):
                if data[j] < theta < data[j+1]:
                    start = j
                if data[j] > theta > data[j+1]:
                    end = j
                    break
            if end - start > 10:
                result = data[start:end:1]
                break
        return result

    def get_params(self, data):
        # 平均绝对值
        MAV = sum([abs(x) for x in data]) / len(data)
        # 归一化处理
        data = [(x - sum(data)/len(data)) / (max(data) - min(data)) for x in data]
        # 过零点数
        ZC = 0
        # 波形长度
        WL = len(data)
        # 斜率变化数
        SSC = 0
        for i in range(len(data) - 1):
            if data[i] * data[i+1] < 0:
                ZC += 1
            if i > 0 and (data[i-1] < data[i] < data[i+1] or data[i+1] < data[i] < data[i-1]):
                SSC += 1
        params = [MAV, ZC, WL, SSC]
        return params

    def train(self):
        train_data = list()
        for row in self.train_row_data:
            train_data.append(self.get_params(row))
        self.model.fit(train_data, self.train_target)
        self.save_train_data()

    def predict(self):
        predict_data = list()
        for row in self.predict_row_data:
            predict_data.append(self.get_params(row))
        y = self.model.predict(predict_data)
        num = 0
        for i in range(len(y)):
            if self.predict_target[i] == y[i]:
                num += 1
        print("准确率：%s %%" % ((num / len(self.predict_target)) * 100))
        self.save_predict_data()

    def save_train_data(self):
        f = open('train_row_data.txt', 'w+')
        for arr in self.train_row_data:
            arr = [str(x) for x in arr]
            f.write(",".join(arr))
            f.write("\n")
        f.close()

        f = open('train_target.txt', 'w+')
        for s in self.train_target:
            f.write(s)
            f.write("\n")
        f.close()

    def save_predict_data(self):
        f = open('predict_row_data.txt', 'w+')
        for arr in self.predict_row_data:
            arr = [str(x) for x in arr]
            f.write(",".join(arr))
            f.write("\n")
        f.close()

        f = open('predict_target.txt', 'w+')
        for s in self.predict_target:
            f.write(s)
            f.write("\n")
        f.close()

    def load_train_data(self):
        f = open('train_row_data.txt', 'r')
        for line in f.readlines():
            arr = [float(x) for x in line.split(",")]
            self.train_row_data.append(arr)
        f.close()
        f = open('train_target.txt', 'r')
        for line in f.readlines():
            self.train_target.append(line.strip())
        f.close()

    def load_predict_data(self):
        f = open('predict_row_data.txt', 'r')
        for line in f.readlines():
            arr = [float(x) for x in line.split(",")]
            self.predict_row_data.append(arr)
        f.close()
        f = open('predict_target.txt', 'r')
        for line in f.readlines():
            self.predict_target.append(line.strip())
        f.close()

    def collect_train(self):
        print("收集训练数据")
        for pose in self.labels:
            print("Action %s" % pose)
            for i in range(self.train_num):
                input("第%s个动作,按回车开始" % i)
                serial_data = list()
                mean_data = list()
                while 1:
                    self.collect_signal(serial_data, mean_data)
                    if self.validate_train_data(pose, serial_data):
                        break

    def collect_predict(self):
        print("收集预测数据")
        for pose in self.labels:
            print("Action %s" % pose)
            for i in range(self.predict_num):
                input("第%s个动作,按回车开始" % i)
                serial_data = list()
                mean_data = list()
                while 1:
                    self.collect_signal(serial_data, mean_data)
                    if self.validate_predict_data(pose, serial_data):
                        break


if __name__ == "__main__":
    pose_data = ["left", "rest", "open"]
    model = EmgModel(pose_data)
    # model.load_train_data()
    model.collect_train()
    model.train()
    # model.load_predict_data()
    model.collect_predict()
    model.predict()
