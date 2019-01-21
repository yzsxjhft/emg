import serial
import numpy
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


class EmgModel:
    def __init__(self, labels):
        self.model = GaussianNB()
        self.labels = labels
        self.serial = None

    def collect_signal(self, serial_data):
        try:
            value = str(self.serial.readline())
            value = value.replace("b", "").replace("\\r\\n", "").replace("\'", "")
            value0 = float(value.split(",")[1])
            value1 = float(value.split(",")[2])
            print([value0, value1])
        except IndexError:
            print("Index Error")
            return
        serial_data.append([value0, value1])

    def get_wave(self, data):
        theta = 90
        start = 0
        end = 0
        result = []
        for i in range(len(data) - 1):
            if sum(data[i]) < theta < sum(data[i+1]):
                start = i
            if sum(data[i]) > theta > sum(data[i+1]):
                end = i
                if end - start > 10:
                    result.append(data[start:end:1])
        return result

    def get_params(self, dataArr):
        params = list()
        arr = list()
        for i in range(len(dataArr[0])):
            arr.append([x[i] for x in dataArr])
        for data in arr:
            if max(data) == min(data):
                return None
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
            params.extend([MAV, ZC, WL, SSC])
        return params

    def predict(self):
        all_data = list()
        all_target = list()
        for pose in self.labels:
            data = self.load_data(pose)
            waves = self.get_wave(data)
            target = [pose] * len(waves)
            all_data += waves
            all_target += target
        # 提取特征值
        params_data = list()
        for i in range(len(all_data)):
            row = all_data[i]
            params = self.get_params(row)
            if params is None:
                all_target.pop(i)
            else:
                params_data.append(params)
        # 训练数据和测试数据分离
        train_data = params_data[0::2]
        train_target = all_target[0::2]
        predict_data = params_data[1::2]
        predict_target = all_target[1::2]
        # score = cross_val_score(self.model, train_data, train_target, cv=5, scoring='accuracy')
        # print(score)
        # print(score.mean())
        # 模型训练
        self.model.fit(train_data, train_target)
        # 模型预测
        y = self.model.predict(predict_data)
        num = 0
        for i in range(len(y)):
            if predict_target[i] == y[i]:
                num += 1

        cm = metrics.confusion_matrix(y, predict_target)
        print("混淆矩阵")
        print(cm)
        print("分类报告")
        cr = metrics.classification_report(y, predict_target)
        print(cr)
        print("准确率：%s %%" % ((num / len(predict_target)) * 100))

    def save_data(self, filename, data):
        f = open(filename, 'w+')
        for d in data:
            f.write(str(d))
            f.write("\n")
        f.close()

    def load_data(self, filename):
        f = open(filename, 'r')
        data = list()
        for line in f.readlines():
            row = line.replace('\n', '').replace("[", "").replace("]", "").split(",")
            row = [float(x) for x in row]
            data.append(row)
        f.close()
        return data

    def collect_data(self):
        self.serial = serial.Serial('COM5', 9600)
        print("收集训练数据")
        for pose in self.labels:
            input("Action %s,按回车开始" % pose)
            serial_data = list()
            for num in range(10000):
                self.collect_signal(serial_data)
            self.save_data(pose, serial_data)


if __name__ == "__main__":
    pose_data = ["left1", "rest1", "open1"]
    model = EmgModel(pose_data)
    # model.collect_data()
    model.predict()
