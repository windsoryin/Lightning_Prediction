import os
import csv
import joblib
import numpy
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import sklearn.metrics as metrics
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def Find_Optimal_Cutoff(TPR, FPR, threshold):
    """
    threshold 一般通过sklearn.metrics里面的roc_curve得到，具体不赘述，可以参考其他资料。
    :param threshold: array, shape = [n_thresholds]
    """
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def plot_AUC(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show(block=True)


# 评分函数
def Score(y, y_pred, index):
    # cm = confusion_matrix(y_true=y, y_pred=y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    # disp.plot()
    # plt.show(block=True)

    FP = 0
    TP = 0
    TN = 0
    FN = 0
    n = 0
    for i in range(0, len(index)):
        if y_pred[n] == 0 and y[index[i]] == 1:
            FN = FN + 1
        if y_pred[n] == 1 and y[index[i]] == 0:
            FP = FP + 1
        if y_pred[n] == 1 and y[index[i]] == 1:
            TP = TP + 1
        if y_pred[n] == 0 and y[index[i]] == 0:
            TN = TN + 1
        n = n + 1
    POD = round((TP / (TP + FN)) * 100.0, 2)
    FAR = round((FP / (FP + TP)) * 100.0, 2)
    return FN, FP, TN, TP, POD, FAR

    # print(f'Fold:{j} on train_data:FN={FN},FP={FP},TP={TP},TN={TN}')
    # print("POD: %.2f%%" % ((TP / (TP + FN)) * 100.0))
    # print("FAR: %.2f%%" % ((FP / (FP + TP)) * 100.0))


# parameter search ()
def Data1_Lightning_Prediction_Model_1(hyperparameters, data1_csv_path, lightning_model_on_data_1_path):
    data = pd.read_csv(f'{data1_csv_path}')
    min_test_FAR_avg = 100
    for hp in hyperparameters:
        # Data1 Lightning Prediction Model 1
        train_PODs = []
        train_FARs = []
        test_PODs = []
        test_FARs = []

        model = EasyEnsembleClassifier(n_estimators=hp['n_estimators'], random_state=hp['random_state'],
                                       base_estimator=AdaBoostClassifier(random_state=42))
        kf = KFold(n_splits=3, shuffle=False)

        for train_index, test_index in kf.split(data):
            train_years = set(data.iloc[train_index]['year'])
            test_years = set(data.iloc[test_index]['year'])
            print(f'train_years:{train_years}')
            print(f'test_years:{test_years}')
            # 将数据集划分为训练集和测试集
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            X_train, X_test = data.iloc[train_index].drop(['flash'], axis=1), data.iloc[test_index].drop(['flash'],
                                                                                                         axis=1)
            y_train, y_test = data.iloc[train_index]['flash'], data.iloc[test_index]['flash']

            model.fit(X_train, y_train)

            probs = model.predict_proba(X_train)  # 得到含有每个样本的概率矩阵
            preds = probs[:, 1]  # 提取出预测为闪电的概率
            fpr, tpr, threshold = metrics.roc_curve(y_train, preds)  # 计算真正率和假正率
            optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
            preds[preds >= optimal_threshold] = 1
            preds[preds < optimal_threshold] = 0
            y_train_pred = preds

            # 计算训练集上评分
            FN, FP, TN, TP, POD, FAR = Score(y_train, y_train_pred, train_index)

            train_PODs.append(POD)
            train_FARs.append(FAR)

            probs = model.predict_proba(X_test)
            preds = probs[:, 1]  # 提取出预测为闪电的概率
            fpr, tpr, threshold = metrics.roc_curve(y_test, preds)  # 计算真正率和假正率
            optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
            preds[preds >= optimal_threshold] = 1
            preds[preds < optimal_threshold] = 0
            y_test_pred = preds

            # 计算测试集上评分
            FN, FP, TN, TP, POD, FAR = Score(y_test, y_test_pred, test_index)
            test_PODs.append(POD)
            test_FARs.append(FAR)

        # 找出循环中最小的FAR，并保存该次循环的模型
        test_FAR_avg = sum(test_FARs) / len(test_FARs)
        if test_FAR_avg < min_test_FAR_avg:
            min_test_FAR_avg = test_FAR_avg
            joblib.dump(filename=lightning_model_on_data_1_path, value=model)
            print("hp['n_estimators'], hp['random_state'] = ", hp['n_estimators'], hp['random_state'],
                  f', min_test_FAR_avg = {min_test_FAR_avg}')


# Add select on Data 1 and Detector train on Data 1
def Detector_train_on_Data_1(lightning_model_path, data1_csv_path, add_select_on_data_1_to_csv_path, detector_on_data_1_model_path):

    # Add select on Data 1
    model = joblib.load(filename=lightning_model_path)
    data = pd.read_csv(data1_csv_path)
    X_data = data.drop(['flash'], axis=1)
    y_data = data['flash']

    y_pred = model.predict(X_data)

    # 合并数组
    data = numpy.column_stack((data, y_pred))
    # 将训练结果写入文件

    fieldnames = ["year", "month", "day", "hour", "doy", "t2m", "sp", "rh", "tp", "flash", "pred_flash", "select"]

    with open(add_select_on_data_1_to_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(data)

    df = pd.read_csv(add_select_on_data_1_to_csv_path)
    df['select'] = df['flash'].eq(df['pred_flash']).map({True: 1, False: 0})
    df.to_csv(add_select_on_data_1_to_csv_path, index=False)

    # Detector train on Data 1
    data = pd.read_csv(add_select_on_data_1_to_csv_path)
    data = data.drop(['pred_flash'], axis=1)

    # Obtain feature names
    feature_name = ['year', 'month', 'day', 'hour', 'doy', 't2m', 'sp', 'rh', 'tp', 'flash', 'select']
    print(feature_name)
    # Select the first 10 columns
    selected_data = data.iloc[:, :10]
    selected_data = np.array(selected_data)
    selected_target = data.iloc[:, 10]
    selected_target = np.array(selected_target)
    print(selected_target)

    X, y = selected_data, selected_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=None)
    evalset = [(X_train, y_train), (X_test, y_test)]

    # define model
    model = xgb.XGBClassifier(max_depth=5, n_estimators=50, silent=True, feature_names=feature_name, num_class=2,
                              booster='gbtree', min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softmax", eval_metric='mlogloss')

    model.fit(X_train, y_train, verbose=True, eval_set=evalset, early_stopping_rounds=10)

    # Prediction
    y_pred = model.predict(X_test)
    y_pred_2 = model.predict(X_train)

    model.save_model(detector_on_data_1_model_path)


# Detector evaluate on Data 2
def Detector_evaluate_on_Data_2(detector_on_data_1_model_path, data2_csv_path):

    # Loading the model
    model = xgb.XGBClassifier()
    model.load_model(detector_on_data_1_model_path)

    data = pd.read_csv(data2_csv_path)
    data['select'] = 1

    selected_data = data.iloc[:, :10]
    selected_data = np.array(selected_data)
    selected_target = data.iloc[:, 10]
    selected_target = np.array(selected_target)

    X, y = selected_data, selected_target

    # Prediction
    y_pred = model.predict(X)

    df = pd.read_csv(data2_csv_path)

    y_pred_series = pd.Series(y_pred)

    df['select'] = y_pred_series
    # 保存修改后的数据保存到原始 CSV 文件
    df.to_csv(data2_csv_path, index=False)

# Data2: Lightning Prediction Model 2
def Lightning_Prediction_Model_2(data2_csv_path, lightning_model_on_data_1_path):
    data = pd.read_csv(data2_csv_path)
    model = joblib.load(filename=lightning_model_on_data_1_path)

    kf = KFold(n_splits=8, shuffle=False)
    j = 1
    for train_index, test_index in kf.split(data):
        train_years = set(data.iloc[train_index]['year'])
        test_years = set(data.iloc[test_index]['year'])
        print(f'train_years:{train_years}')
        print(f'test_years:{test_years}')
        # 将数据集划分为训练集和测试集
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        X_train, X_test = data.iloc[train_index].drop(['flash'], axis=1), data.iloc[test_index].drop(['flash'], axis=1)
        y_train, y_test = data.iloc[train_index]['flash'], data.iloc[test_index]['flash']
        X_train = X_train.drop(columns=['select'])
        X_test = X_test.drop(columns=['select'])

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_train)  # 得到含有每个样本的概率矩阵
        preds = probs[:, 1]  # 提取出预测为闪电的概率
        fpr, tpr, threshold = metrics.roc_curve(y_train, preds)  # 计算真正率和假正率
        optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
        preds[preds >= optimal_threshold] = 1
        preds[preds < optimal_threshold] = 0
        y_train_pred = preds

        # 计算训练集上评分
        FN, FP, TN, TP, POD, FAR = Score(y_train, y_train_pred, train_index)
        print(f'Fold:{j} on train_data_2:FN={FN},FP={FP},TP={TP},TN={TN}')
        print("POD: %.2f%%" % ((TP / (TP + FN)) * 100.0))
        print("FAR: %.2f%%" % ((FP / (FP + TP)) * 100.0))

        probs = model.predict_proba(X_test)
        preds = probs[:, 1]  # 提取出预测为闪电的概率
        fpr, tpr, threshold = metrics.roc_curve(y_test, preds)  # 计算真正率和假正率
        optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
        preds[preds >= optimal_threshold] = 1
        preds[preds < optimal_threshold] = 0
        y_test_pred = preds

        # 计算测试集上评分
        FN, FP, TN, TP, POD, FAR = Score(y_test, y_test_pred, test_index)
        print(f'Fold:{j} on test_data_2:FN={FN},FP={FP},TP={TP},TN={TN}')
        print("POD: %.2f%%" % ((TP / (TP + FN)) * 100.0))
        print("FAR: %.2f%%" % ((FP / (FP + TP)) * 100.0))
        j = j + 1

# overall metric


def main():
    # # ————————————————————————————————————————————————————————————————————————————————————————————————————
    path = r'D:\桌面\不同分辨率数据\10min\实验数据\30km'
    # # # 处理数据部分：将 2018/01/01 00:00:00 格式的date分为5列
    # # for file in os.listdir(path):
    # #     if file.endswith('.csv'):
    # #         df = pd.read_csv(os.path.join(path, file))
    # #         df['date'] = pd.to_datetime(df['date'])
    # #         df['year'] = df['date'].dt.year
    # #         df['month'] = df['date'].dt.month
    # #         df['day'] = df['date'].dt.day
    # #         df['hour'] = df['date'].dt.hour
    # #         df['doy'] = df['date'].dt.dayofyear
    # #         df = df[['year', 'month', 'day', 'hour', 'doy', 't2m', 'sp', 'rh', 'tp', 'flash']]
    # #         df.to_csv(os.path.join(path, file), index=False)
    #
    # dataset split into 1 and 2
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()

    for file in os.listdir(path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            df1 = df[(df['year'] >= 2018) & (df['year'] <= 2020)]
            data1 = pd.concat([data1, df1], ignore_index=True)
            df2 = df[(df['year'] >= 2021) & (df['year'] <= 2022)]
            data2 = pd.concat([data2, df2], ignore_index=True)
    data1 = data1.sort_values(by=['year', 'month', 'day', 'hour'])
    data2 = data2.sort_values(by=['year', 'month', 'day', 'hour'])
    data1.to_csv('../data/data1.csv', index=False)
    data2.to_csv('../data/data2.csv', index=False)

    # ————————————————————————————————————————————————————————————————————————————————————————————————————

    data1_csv_path = '../data/data1.csv'
    data2_csv_path = '../data/data2.csv'
    lightning_model_on_data_1_path = '../model/lightning_on_data_1.h5'
    add_select_on_data_1_to_csv_path = '../data/data1_add_select.csv'
    detector_on_data_1_model_path = '../model/detector_model_on_data_1.h5'
    hyperparameters = [
        {'n_estimators': 50, 'random_state': 42},
        {'n_estimators': 35, 'random_state': 42},
        {'n_estimators': 25, 'random_state': 42}
    ]
    Data1_Lightning_Prediction_Model_1(hyperparameters, data1_csv_path, lightning_model_on_data_1_path)

    Detector_train_on_Data_1(lightning_model_on_data_1_path, data1_csv_path, add_select_on_data_1_to_csv_path, detector_on_data_1_model_path)

    Detector_evaluate_on_Data_2(detector_on_data_1_model_path, data2_csv_path)

    Lightning_Prediction_Model_2(data2_csv_path, lightning_model_on_data_1_path)


if __name__ == "__main__":
    main()


