from sklearn import metrics
from Knn import Knn
import os
import numpy as np


class EmgModel:
    def __init__(self, labels):
        self.model = Knn(k=5)
        self.labels = labels
        self.all_data = list()
        self.all_target = list()
        for pose in labels:
            label = labels[pose]
            data = self.load_data(label)
            target = [pose] * len(data)
            self.all_data += data
            self.all_target += target

    def run(self):
        # 训练数据和测试数据分离
        train_data = self.all_data[0::2]
        train_target = self.all_target[0::2]
        predict_data = self.all_data[1::2]
        predict_target = self.all_target[1::2]

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

    def load_data(self, label):
        parent_dir = "data/%s" % label
        res = []
        for file in os.listdir(parent_dir):
            filename = "%s/%s" % (parent_dir, file)
            f = open(filename, 'r')
            data = []
            for line in f.readlines():
                row = line.replace('\n', '').replace("[", "").replace("]", "").split(",")
                row = [float(x) for x in row]
                data.append(row)
            f.close()
            data = np.array(data).T
            res.append(data)
        return res


if __name__ == "__main__":
    pose_data = {1: "left", 2: "right", 3: "rest", 4: "open"}
    model = EmgModel(pose_data)
    model.run()
