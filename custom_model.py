from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier

from flaml import AutoML
from flaml import tune
import xgboost as xgb

import joblib



class custom_model():
    def __init__(self, args):
        self.args = args
        self.model= self.build()

    def build(self):
        if self.args.model == 'flaml':
            model = AutoML()
            self.setting= {
            "time_budget": 50,  # 总时间上限(单位秒)
            "metric": 'roc_auc',
            # 候选可以是: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
                # 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
            "task": 'classification',  # 任务类型
            "estimator_list": ['xgboost'],
            "log_file_name": 'airlines_experiment.log',  # flaml日志文件
            "log_training_metric": True,  # whether to log training metric
            "keep_search_state": True,  # needed if you want to keep the cross validation information
            "seed": 7654321,  # 随机种子
            # "ensemble": 'True',
            "eval_method":'cv',
            "n_splits":3,
            }
        if self.args.model=='easy_ensemble':
            model = EasyEnsembleClassifier(n_estimators=self.args.n_estimators, random_state=self.args.random_state,
                                           base_estimator=AdaBoostClassifier(random_state=42))
        if self.args.model=='xgboost':
            model = xgb.XGBClassifier(max_depth=5, n_estimators=50, silent=True,  num_class=2,
                              booster='gbtree', min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                              objective="multi:softmax", eval_metric='mlogloss')

        return model

    def model(self):
        return self.model

    def fit(self,data,target):
        if self.args.model == 'flaml':
            self.model.fit(X_train=data, y_train=target,**self.setting,
                           #           custom_hp={
                           #               "xgboost": {
                           #                   "n_estimators": {
                           #                       "domain": tune.lograndint(lower=1, upper=70),
                           #                       "low_cost_init_value": 1,
                           #                   },
                           #                   "max_leaves":{
                           #                       "domain": tune.lograndint(lower=1, upper=60),
                           #                       "low_cost_init_value": 1,
                           #                   },
                           #               },
                           #               "rf": {
                           #                   "max_leaves": {
                           #                       "domain": None,  # disable search
                           #                   },
                           #               },
                           #               "lgbm": {
                           #                   "subsample": {
                           #                       "domain": tune.uniform(lower=0.1, upper=1.0),
                           #                       "init_value": 1.0,
                           #                   },
                           #                   "subsample_freq": {
                           #                       "domain": 1,  # subsample_freq must > 0 to enable subsample
                           #                   },
                           #               },
                           #           }
                           )
        if self.args.model == 'easy_ensemble':
            self.model.fit(data,target)

    def predict(self,data):
        return self.model.predict(data)

    def predict_proba(self,data):
        return self.model.predict_proba(data)

    def save_model(self, path):
        joblib.dump(filename=path, value=self.model)
    #