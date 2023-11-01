from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import AdaBoostClassifier
from flaml.automl.model import SKLearnEstimator
from flaml.automl.task.task import CLASSIFICATION

from flaml import AutoML
from flaml import tune
import xgboost as xgb

import joblib


class MyEasyEnsemble(SKLearnEstimator):
    def __init__(self, task='binary', **config):
        '''Constructor

        Args:
            task: A string of the task type, one of
                'binary', 'multiclass', 'regression'
            config: A dictionary containing the hyperparameter names
                and 'n_jobs' as keys. n_jobs is the number of parallel threads.
        '''

        super().__init__(task, **config)

        '''task=binary or multi for classification task'''
        if task in CLASSIFICATION:
            from imblearn.ensemble import EasyEnsembleClassifier
            self.estimator_class = EasyEnsembleClassifier
        else:
            from imblearn.ensemble import EasyEnsembleClassifier
            self.estimator_class = EasyEnsembleClassifier

    @classmethod
    def search_space(cls, data_size, task):
        '''[required method] search space

        Returns:
            A dictionary of the search space.
            Each key is the name of a hyperparameter, and value is a dict with
                its domain (required) and low_cost_init_value, init_value,
                cat_hp_cost (if applicable).
                e.g.,
                {'domain': tune.randint(lower=1, upper=10), 'init_value': 1}.
        '''
        space = {
            'n_estimators': {'domain': tune.randint(lower=40, upper=200), 'init_value': 4,
                         'low_cost_init_value': 4},
            'random_state': {'domain': tune.randint(lower=40, upper=200), 'init_value': 4,
                             'low_cost_init_value': 4},
        }
        return space



class custom_model():
    def __init__(self, args):
        self.args = args
        self.model= self.build()

    def build(self):
        if self.args.model == 'flaml':
            model = AutoML()
            self.setting= {
            "time_budget": self.args.time_budget,  # 总时间上限(单位秒)
            "metric": 'f1',
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
            "n_splits":self.args.n_split,
            }
        if self.args.model=='easy_ensemble':
            model = AutoML()
            self.settings = {
                "time_budget": self.args.time_budget,  # total running time in seconds
                "metric": 'log_loss',
                "estimator_list": ['easy_ensemble'],  # list of ML learners
                "task": 'classification',  # task type
                "log_file_name": 'airlines_experiment_custom_learner.log',  # flaml log file
                "log_training_metric": True,  # whether to log training metric
                "keep_search_state": True,  # needed if you want to keep the cross validation information
                "seed": 7654321,  # 随机种子
                # "ensemble": 'True',
                "eval_method": 'cv',
                "n_splits": self.args.n_split,
            }
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
                                     custom_hp={
                                         "xgboost": {
                                             "n_estimators": {
                                                 "domain": tune.lograndint(lower=1, upper=70),
                                                 "low_cost_init_value": 1,
                                             },
                                             "max_leaves":{
                                                 "domain": tune.lograndint(lower=1, upper=70),
                                                 "low_cost_init_value": 1,
                                             },
                                         },
                                         "rf": {
                                             "max_leaves": {
                                                 "domain": None,  # disable search
                                             },
                                         },
                                         "lgbm": {
                                             "subsample": {
                                                 "domain": tune.uniform(lower=0.1, upper=1.0),
                                                 "init_value": 1.0,
                                             },
                                             "subsample_freq": {
                                                 "domain": 1,  # subsample_freq must > 0 to enable subsample
                                             },
                                         },
                                     }
                           )
        if self.args.model == 'easy_ensemble':
            self.model.add_learner(learner_name='easy_ensemble', learner_class=MyEasyEnsemble)
            self.model.fit(X_train=data, y_train=target,**self.settings)

    def predict(self,data):
        return self.model.predict(data)

    def predict_proba(self,data):
        return self.model.predict_proba(data)

    def save_model(self, path):
        joblib.dump(filename=path, value=self.model)

    def feature_importance(self):

        return self.model.feature_names_in_, self.model.feature_importances_