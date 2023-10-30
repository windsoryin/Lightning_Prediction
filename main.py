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
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,precision_recall_curve
import matplotlib.pyplot as plt
from flaml import AutoML
from flaml import tune
import matplotlib.pyplot as plt
import argparse

from custom_model import custom_model

#sss
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
def Score(y, y_pred, index,title):
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    ax_1=plt.figure().subplots()
    ax_1.set(title=title)
    disp.plot(ax=ax_1)
    plt.show(block=True)

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
    POD = 0 if (TP + FN)==0 else round((TP / (TP + FN)) * 100.0, 2)
    FAR = 0 if (FP + TP)==0 else round((FP / (FP + TP)) * 100.0, 2)
    return FN, FP, TN, TP, POD, FAR

    # print(f'Fold:{j} on train_data:FN={FN},FP={FP},TP={TP},TN={TN}')
    # print("POD: %.2f%%" % ((TP / (TP + FN)) * 100.0))
    # print("FAR: %.2f%%" % ((FP / (FP + TP)) * 100.0))




# parameter search ()
def Data1_Lightning_Prediction_Model_1(args,hyperparameters, data1_csv_path, lightning_model_on_data_1_path):
    # Data1 Lightning Prediction Model 1
    # k-fold cross validation

    data = pd.read_csv(f'{data1_csv_path}')
    # flash_num=data['flash'].sum()
    min_test_FAR_avg = 100

    X_data = data.drop(['flash'], axis=1)
    y_data = data['flash']
    model = custom_model(args)
    model.fit(X_data, y_data)

    probs = model.predict_proba(X_data)  # 得到含有每个样本的概率矩阵
    preds = probs[:, 1]  # 提取出预测为闪电的概率
    plot_AUC(model, X_data, y_data)
    fpr, tpr, threshold = metrics.roc_curve(y_data, preds)  # 计算真正率和假正率
    optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
    optimal_threshold = optimal_threshold+0.4
    preds[preds >= optimal_threshold] = 1
    preds[preds < optimal_threshold] = 0
    y_train_pred = preds

    train_index=y_data.index.values
    # 计算训练集上评分
    FN, FP, TN, TP, POD, FAR = Score(y_data, y_train_pred, train_index, 'train results')

    # sava the model
    model.save_model(lightning_model_on_data_1_path)
    np.save('thresh.npy', optimal_threshold)


    # # k-fold CV using POD and FAR as eval metric
    # kf = KFold(n_splits=3, shuffle=False)
    #
    # for train_index, test_index in kf.split(data):
    #     train_PODs = []
    #     train_FARs = []
    #     test_PODs = []
    #     test_FARs = []
    #
    #     train_data = data.iloc[train_index]
    #     test_data = data.iloc[test_index]
    #
    #     X_train, X_test = data.iloc[train_index].drop(['flash'], axis=1), data.iloc[test_index].drop(['flash'],
    #                                                                                                  axis=1)
    #     y_train, y_test = data.iloc[train_index]['flash'], data.iloc[test_index]['flash']
    #     # use self defined model
    #     model=custom_model(args)
    #     model.fit(X_train, y_train)
    #     y_train_pred = model.predict(X_train)
    #     FN, FP, TN, TP, POD, FAR = Score(y_train, y_train_pred, train_index, 'train results')
    #
    #     kf = KFold(n_splits=3, shuffle=False)
    #
    #     for train_index, test_index in kf.split(data):
    #
    #         train_data = data.iloc[train_index]
    #         test_data = data.iloc[test_index]
    #
    #         X_train, X_test = data.iloc[train_index].drop(['flash'], axis=1), data.iloc[test_index].drop(['flash'],
    #                                                                                                      axis=1)
    #         y_train, y_test = data.iloc[train_index]['flash'], data.iloc[test_index]['flash']
    #
    #
    #
    #
    #
    #         probs = model.predict_proba(X_train)  # 得到含有每个样本的概率矩阵
    #         preds = probs[:, 1]  # 提取出预测为闪电的概率
    #         plot_AUC(model, X_train, y_train)
    #         fpr, tpr, threshold = metrics.roc_curve(y_train, preds)  # 计算真正率和假正率
    #         optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
    #         optimal_threshold=optimal_threshold
    #         preds[preds >= optimal_threshold] = 1
    #         preds[preds < optimal_threshold] = 0
    #         y_train_pred = preds
    #
    #         # 计算训练集上评分
    #         FN, FP, TN, TP, POD, FAR = Score(y_train, y_train_pred, train_index,'train results')
    #
    #         train_PODs.append(POD)
    #         train_FARs.append(FAR)
    #
    #         y_test_pred=model.predict(X_test)
    #         FN, FP, TN, TP, POD, FAR = Score(y_test, y_test_pred, test_index,'test results')
    #         # probs = model.predict_proba(X_test)
    #         # preds = probs[:, 1]  # 提取出预测为闪电的概率
    #         # # test dataset 不能调整，只能采用train data 得到的阈值
    #         # plot_AUC(model, X_test, y_test)
    #         # # fpr, tpr, threshold = metrics.roc_curve(y_test, preds)  # 计算真正率和假正率
    #         # # optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
    #         # optimal_threshold=0.5
    #         # preds[preds >= optimal_threshold] = 1
    #         # preds[preds < optimal_threshold] = 0
    #         # y_test_pred = preds
    #
    #         # 计算测试集上评分
    #         # FN, FP, TN, TP, POD, FAR = Score(y_test, y_test_pred, test_index,'test results')
    #         test_PODs.append(POD)
    #         test_FARs.append(FAR)
    #         # plt.barh(model.model.estimator.feature_name_, model.model.estimator.feature_importances_)
    #     # 找出循环中最小的FAR，并保存该次循环的模型
    #     test_FAR_avg = sum(test_FARs) / len(test_FARs)
    #     if test_FAR_avg < min_test_FAR_avg:
    #         min_test_FAR_avg = test_FAR_avg
    #         joblib.dump(filename=lightning_model_on_data_1_path, value=custom_model.model)
    #         print("hp['n_estimators'], hp['random_state'] = ", hp['n_estimators'], hp['random_state'],
    #               f', min_test_FAR_avg = {min_test_FAR_avg}')
    #         np.save('thresh.npy', optimal_threshold)



# Add select on Data 1 and Detector train on Data 1
def Detector_train_on_Data_1(lightning_model_path, data1_csv_path, add_select_on_data_1_to_csv_path, detector_on_data_1_model_path):
    optimal_threshold=np.load('thresh.npy')
    # Add select on Data 1
    model = joblib.load(filename=lightning_model_path)
    data = pd.read_csv(data1_csv_path)
    X_data = data.drop(['flash'], axis=1)
    y_data = data['flash']

    y_pred_data1 = model.predict_proba(X_data)
    y_pred_data1 =(y_pred_data1[:, 1] >= optimal_threshold).astype(int) # 提取出预测为闪电的概率

    # Predictor 1 results on Data 1
    cm = confusion_matrix(y_true=y_data, y_pred=y_pred_data1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    disp.plot()
    plt.show(block=True)
    # 合并数组
    data = data.assign(pred_flash=y_pred_data1)
    # 将训练结果写入文件
    data.to_csv(add_select_on_data_1_to_csv_path, index=False)

    # generate 'select' column
    # predict == true: 0
    # predict != true: 1  (anomaly value)
    df = pd.read_csv(add_select_on_data_1_to_csv_path)
    df['select'] = df['flash'].eq(df['pred_flash']).map({True: 0, False: 1})
    df.to_csv(add_select_on_data_1_to_csv_path, index=False)

    # Detector train on Data 1
    data = pd.read_csv(add_select_on_data_1_to_csv_path)
    data = data.drop(['pred_flash'], axis=1)

    # Select columns except {'select','flash'}
    X_Train = data.iloc[:, :-2]
    X_Train = np.array(X_Train)
    y_target = data.iloc[:, -1]
    y_target = np.array(y_target)

    # X, y = selected_data, selected_target
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, shuffle=None)
    # evalset = [(X_train, y_train), (X_test, y_test)]

    model = AutoML()

    # 参数设定
    settings = {
        "time_budget": 40,  # 总时间上限(单位秒)
        "metric": 'roc_auc',
        # 候选可以是: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
        "task": 'classification',  # 任务类型
        # "estimator_list": ['xgboost'],
        "log_file_name": 'airlines_experiment.log',  # flaml日志文件
        "log_training_metric": True,  # whether to log training metric
        "keep_search_state": True,  # needed if you want to keep the cross validation information
        "seed": 7654321,  # 随机种子
        # "ensemble": 'True',
        "eval_method": 'cv',
        "n_splits": 3,
    }
    model.fit(X_train=X_Train, y_train=y_target, **settings)


    # Prediction
    probs = model.predict_proba(X_Train)  # 得到含有每个样本的概率矩阵
    preds = probs[:, 1]  # 提取出预测为闪电的概率
    plot_AUC(model, X_Train, y_target)
    fpr, tpr, threshold = metrics.roc_curve(y_target, preds)  # 计算真正率和假正率
    optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
    optimal_threshold = optimal_threshold
    preds[preds >= optimal_threshold] = 1
    preds[preds < optimal_threshold] = 0

    cm = confusion_matrix(y_true=y_target, y_pred=preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    disp.plot()
    plt.show(block=True)


    ###################################
    # step 3: evaluate Detector on Data 1

    # set the column 'select' with the prediction result
    data = pd.read_csv(data1_csv_path)
    X_data = data.drop(['flash'], axis=1).values
    y_data = data['flash'].values

    # Prediction
    y_pred_slect = model.predict(X_data)
    y_pred_series = pd.Series(y_pred_slect)
    data['select'] = y_pred_series
    print(sum(y_pred_slect))

    # check results on Data 1 after removing detector
    y_data=y_data[(y_pred_slect==0)]
    y_pred_data1=y_pred_data1[(y_pred_slect==0)]
    index=np.arange(len(y_data))
    FN, FP, TN, TP, POD, FAR = Score(y_data, y_pred_data1, index, 'result after Detector')

    # Save detector model
    joblib.dump(filename=detector_on_data_1_model_path, value=model)

    print('step 2 finish')
# Predictor 1 evaluate on Data 2 Directly


# Detector evaluate on Data 2
def Detector_evaluate_on_Data_2(detector_on_data_1_model_path, data2_csv_path):

    # Loading the model
    model=joblib.load(detector_on_data_1_model_path)
    data = pd.read_csv(data2_csv_path)

    selected_data = data.iloc[:, :-1]
    selected_data = np.array(selected_data)

    # Prediction
    y_pred_select = model.predict(selected_data)

    df = pd.read_csv(data2_csv_path)

    y_pred_series = pd.Series(y_pred_select)

    df['select'] = y_pred_series
    # 保存修改后的数据保存到原始 CSV 文件
    df.to_csv(data2_csv_path, index=False)

# Data2: Lightning Prediction Model 2
def Lightning_Prediction_Model_2(data2_csv_path, lightning_model_on_data_1_path):
    optimal_threshold=np.load('thresh.npy')

    data = pd.read_csv(data2_csv_path)
    model = joblib.load(filename=lightning_model_on_data_1_path)
    X_data = data.iloc[:,:-2]
    y_data = data['flash']

    y_pred_data1 = model.predict_proba(X_data)
    y_pred_data1 = (y_pred_data1[:, 1] >= optimal_threshold).astype(int)  # 提取出预测为闪电的概率

    select_data=data['select']
    print(sum(select_data))
    # Predictor 1 results on Data 1
    cm = confusion_matrix(y_true=y_data, y_pred=y_pred_data1)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    disp.plot()
    plt.show(block=True)
    # After Detector remove anomalies
    cm = confusion_matrix(y_true=y_data[select_data==0], y_pred=y_pred_data1[select_data==0])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['1', '2'])
    disp.plot()
    plt.show(block=True)

    best_model=model.model.estimator

    ################################
    # k-fold
    train_PODs = []
    train_FARs = []
    test_PODs = []
    test_FARs = []
    kf = KFold(n_splits=2, shuffle=False)
    j = 1
    for train_index, test_index in kf.split(data):

        # 将数据集划分为训练集和测试集
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        X_train, X_test = data.iloc[train_index].drop(['flash'], axis=1), data.iloc[test_index].drop(['flash'], axis=1)
        y_train, y_test = data.iloc[train_index]['flash'], data.iloc[test_index]['flash']
        X_train = X_train.drop(columns=['select'])
        X_test = X_test.drop(columns=['select'])

        best_model.fit(X_train,y_train)
        probs = best_model.predict_proba(X_train)  # 得到含有每个样本的概率矩阵
        preds = probs[:, 1]  # 提取出预测为闪电的概率
        plot_AUC(best_model, X_train, y_train)
        fpr, tpr, threshold = metrics.roc_curve(y_train, preds)  # 计算真正率和假正率
        optimal_threshold, point = Find_Optimal_Cutoff(tpr, fpr, threshold)  # 使用约登指数计算最佳阈值
        optimal_threshold = optimal_threshold+0.1
        preds[preds >= optimal_threshold] = 1
        preds[preds < optimal_threshold] = 0
        y_train_pred = preds


        select_train, select_test = data.iloc[train_index]['select'], data.iloc[test_index]['select']

        # 计算训练集上评分
        y_train=y_train[(select_train==0)]
        y_train_pred=y_train_pred[(select_train==0)]
        train_index=train_index[(select_train==0)]
        FN, FP, TN, TP, POD, FAR = Score(y_train, y_train_pred, train_index,'train reusults')
        print(f'Fold:{j} on train_data_2:FN={FN},FP={FP},TP={TP},TN={TN}')
        print("POD: %.2f%%" % ((TP / (TP + FN)) * 100.0))
        print("FAR: %.2f%%" % ((FP / (FP + TP)) * 100.0))
        train_PODs.append(POD)
        train_FARs.append(FAR)

        probs = best_model.predict_proba(X_test)
        preds = probs[:, 1]  # 提取出预测为闪电的概率
        plot_AUC(best_model, X_test, y_test)
        preds[preds >= optimal_threshold] = 1
        preds[preds < optimal_threshold] = 0
        y_test_pred = preds

        # # 计算测试集上评分
        y_test=y_test[(select_test==0)]
        y_test_pred=y_test_pred[(select_test==0)]
        test_index=test_index[(select_test==0)]
        FN, FP, TN, TP, POD, FAR = Score(y_test, y_test_pred, test_index,'test results ')
        test_PODs.append(POD)
        test_FARs.append(FAR)
        print(f'Fold:{j} on test_data_2:FN={FN},FP={FP},TP={TP},TN={TN}')
        print("POD: %.2f%%" % POD)
        print("FAR: %.2f%%" % FAR)
        j = j + 1

    test_FAR_avg = sum(test_FARs) / len(test_FARs)
    print('average test far:', )

# overall metric

if __name__ == "__main__":
    # # ————————————————————————————————————————————————————————————————————————————————————————————————————
    path = r'.\30km'
    station_list=['lfs','sek','wlp'] # station list

    # dataset split into 1 and 2
    data1 = pd.DataFrame()
    data2 = pd.DataFrame()

    for file in os.listdir(path):
        if file.endswith('.csv'):
            if any(station in file for station in station_list):
                df = pd.read_csv(os.path.join(path, file))
                df1 = df[(df['year'] >= 2018) & (df['year'] <= 2020)]
                data1 = pd.concat([data1, df1], ignore_index=True)
                df2 = df[(df['year'] >= 2021) & (df['year'] <= 2022)]
                data2 = pd.concat([data2, df2], ignore_index=True)

    data1 = data1.sort_values(by=['year', 'month', 'day', 'hour'])
    select_columns=['year', 'month', 'day', 'hour', 'doy', 't2m', 'sp', 'rh', 'tp']
    # data1[select_columns]=(data1[select_columns]-data1[select_columns].mean())/data1[select_columns].std()
    data2 = data2.sort_values(by=['year', 'month', 'day', 'hour'])
    # data2[select_columns]=(data2[select_columns]-data2[select_columns].mean())/data2[select_columns].std()


    # delete rows and columns
    delete_columns=['month','day']
    data1=data1.drop(columns=delete_columns)
    data2=data2.drop(columns=delete_columns)

    # add past lightning as input
    ######################################
    data1.reset_index(inplace=True, drop=True)
    data2.reset_index(inplace=True, drop=True)

    past_flash1= data1.iloc[:-1,:]
    past_flash2= data2.iloc[:-1,:]

    data1=data1.drop(data1.index[[0]])
    data2=data2.drop(data1.index[[0]])
    data1.reset_index(inplace=True, drop=True)
    data2.reset_index(inplace=True, drop=True)
    #
    data1['past_flash']=past_flash1['flash']
    data2['past_flash'] = past_flash2['flash']
    columns = list(data1)
    # move the column to head of list using index, pop and insert
    columns.insert(-1, columns.pop(columns.index('past_flash')))
    # use loc to reorder
    data1 = data1.loc[:, columns]
    data2 = data2.loc[:, columns]
    ######################################

    data1.to_csv('./data/data1.csv', index=False)
    data2.to_csv('./data/data2.csv', index=False)

    # ————————————————————————————————————————————————————————————————————————————————————————————————————

    data1_csv_path = './data/data1.csv'
    data2_csv_path = './data/data2.csv'
    lightning_model_on_data_1_path = './model/lightning_on_data_1.h5'
    add_select_on_data_1_to_csv_path = './data/data1_add_select.csv'
    detector_on_data_1_model_path = './model/detector_model_on_data_1.h5'
    hyperparameters = [
        {'n_estimators': 50, 'random_state': 42},
    ]

    #########################################
    # parameter definition

    parser = argparse.ArgumentParser(description='Lightning Prediction')

    parser.add_argument('--model', type=str, required=False, default='flaml',
                        help='model name')
    # easy ensemble parameters
    parser.add_argument('--n_estimators', type=int, required=False, default=40, help='n_estimators')
    parser.add_argument('--random_state', type=int, required=False, default=24, help='random_state')

    args = parser.parse_args()
    #########################################


    # step 1: train predictor 1 on data 1
    # Data1_Lightning_Prediction_Model_1(args,hyperparameters, data1_csv_path, lightning_model_on_data_1_path)
    # step 2: train Detector 1 on Data 1
    # Detector_train_on_Data_1(lightning_model_on_data_1_path, data1_csv_path, add_select_on_data_1_to_csv_path, detector_on_data_1_model_path)
    # Step 3: use Detector remove anomaly value on Data 2
    Detector_evaluate_on_Data_2(detector_on_data_1_model_path, data2_csv_path)
    # Predict on Data 2
    Lightning_Prediction_Model_2(data2_csv_path, lightning_model_on_data_1_path)


