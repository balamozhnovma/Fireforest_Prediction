import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_error

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score

import mlflow
import pickle
X_train = pd.read_csv('train.csv').drop('area', axis = 1)
y_train = pd.read_csv('train.csv')['area']
X_test = pd.read_csv('test.csv').drop('area', axis = 1)
y_test = pd.read_csv('test.csv')['area']

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('Fireforest')

with mlflow.start_run():
    with mlflow.start_run(run_name = 'Random Forest', nested = True):
        """Random Forest"""
        rf_n_estimators = 100
        min_samples_split = 2
        min_samples_leaf = 5
        print("Обучим Random Forest")
        rf_model = RandomForestRegressor(n_estimators=rf_n_estimators,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        cv_score = cross_val_score(rf_model, X_train, y_train, cv = 5, scoring = 'neg_root_mean_squared_error')

        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f'RMSE = {rmse:.2f}')
        print(f'MAE = {mae:.2f}')
        print(f'CV score = {-1*cv_score.mean():.2f}')

        print("Значение RMSE аномально высокие из-за выбросов. Значение МАЕ хорошо для больших пожаров, но плохо для маленьких или нулевых.")
        print()
        print("Попробуем убрать выбросы и посмотреть на значения ошибок")

        X_train_filtered = pd.read_csv('train_filtered.csv').drop('area', axis = 1)
        y_train_filtered = pd.read_csv('train_filtered.csv')['area']
        X_test_filtered = pd.read_csv('test_filtered.csv').drop('area', axis=1)
        y_test_filtered = pd.read_csv('test_filtered.csv')['area']

        rf_model = RandomForestRegressor(n_estimators = rf_n_estimators,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        rf_model.fit(X_train_filtered, y_train_filtered)
        y_pred_2 = rf_model.predict(X_test_filtered)

        cv_score_2 = cross_val_score(rf_model, X_train_filtered, y_train_filtered, cv = 5, scoring = 'neg_root_mean_squared_error')

        rf_rmse_2 = root_mean_squared_error(y_test_filtered, y_pred_2)
        rf_mae_2 = mean_absolute_error(y_test_filtered, y_pred_2)

        signature = mlflow.models.infer_signature(X_train_filtered, y_pred_2)

        mlflow.log_param('model_type', 'Random Forest')
        mlflow.log_param('n_estimators', rf_n_estimators)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_metric('RF rmse', rf_rmse_2)
        mlflow.log_metric('RF MAE', rf_mae_2)
        mlflow.sklearn.log_model(rf_model, 'model', signature = signature, input_example = X_train)

        print("Обучим Random Forest после обработки выбросов")
        print(f'RMSE = {rf_rmse_2:.2f}')
        print(f'MAE = {rf_mae_2:.2f}')
        print(f'CV score = {-1*cv_score_2.mean():.2f}')

        #График важности фичей
        feature_imp = pd.DataFrame({
            'feature': X_train_filtered.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending = False)

        plt.figure(figsize=(10, 8))
        sns.barplot(data = feature_imp.head(10), x = 'importance', y = 'feature',alpha = 0.8)
        plt.title('График важности фичей')
        plt.show()
        plt.savefig("rf_feature_imp.png")
        plt.close()

        mlflow.log_artifact("rf_feature_imp.png", "rf_hist")

        with open('model_weights/rf_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        print("RMSE и MAE стали значительно ниже, но недостаточно (из-за большого количества нулей в датасете).  Топ фичи - это температура и DC, логично лидируют, так как сухость/жара провоцируют пожары")
        print()
    with mlflow.start_run(run_name = 'Gradient Boosting', nested = True):
        print("Воспользуемся градиентным бустингом **XGBoost**")

        xgb_n_estimators = 100
        learning_rate = 0.1
        reg_alpha = 2
        reg_lambda = 1.5
        xgb_model = xgb.XGBRegressor(n_estimators = xgb_n_estimators,
                                 learning_rate = learning_rate,
                                 reg_alpha = reg_alpha,
                                 reg_lambda = reg_lambda,
                                 verbosity = 0,
                                 eval_metric = 'rmse',
                                 random_state = 42)
        xgb_model.fit(X_train_filtered, y_train_filtered)
        y_pred_2 = xgb_model.predict(X_test_filtered)

        cv_score = cross_val_score(xgb_model, X_train_filtered, y_train_filtered, cv = 5, scoring = 'neg_root_mean_squared_error')

        xgb_rmse = root_mean_squared_error(y_test_filtered, y_pred_2)
        xgb_mae = mean_absolute_error(y_test_filtered, y_pred_2)

        signature = mlflow.models.infer_signature(X_train_filtered, y_pred_2)

        mlflow.log_param('model_type', 'XGBoost')
        mlflow.log_param('n_estimators', xgb_n_estimators)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('reg_alpha', reg_alpha)
        mlflow.log_param('reg_lambda', reg_lambda)
        mlflow.log_metric('XGBoost rmse', xgb_rmse)
        mlflow.log_metric('XGBoost MAE', xgb_mae)
        mlflow.sklearn.log_model(xgb_model, 'XGBoost model', signature = signature, input_example =X_train)

        with open('model_weights/xgb_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        print(f'RMSE = {xgb_rmse:.2f}')
        print(f'MAE = {xgb_mae:.2f}')
        print(f'CV score {-1*cv_score.mean():.2f}')

    #График важности фичей
    feature_imp = pd.DataFrame({
        'feature': X_train_filtered.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending = False)

    plt.figure(figsize=(10, 8))
    sns.barplot(data = feature_imp.head(10), x = 'importance', y = 'feature',alpha = 0.8)
    plt.title('График важности фичей')
    plt.show()
    plt.savefig('xgb_feature_imp.png')
    plt.close()

    print("Он показал результаты немного хуже, чем Random Forest. Скорее всего, из-за малого количества решающих пней в градиентном бустинге. Также для него наиболее важными оказались другие фичи - дни недели и месяц. Подберем гиперпараметры, чтобы улучшить значения метрик")
    print()
    
    # 4. **Подбор гиперпараметров**
    
    print("Воспользуемся библиотекой **Optuna** для точного подбора гиперпараметров")
    
    import optuna
    from optuna.samplers import TPESampler
    
    def optimize_xgboost(trial, X_train, y_train):
        with mlflow.start_run(nested = True):
          params = {
              'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
              'max_depth': trial.suggest_int('max_depth', 2, 10),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
              'subsample': trial.suggest_float('subsample', 0.2, 1.0),
              'reg_alpha': trial.suggest_float('reg_alpha', 1, 10),
              'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
              'verbosity': 0,
              'eval_metric': 'rmse',
              'random_state': 42
          }

          model = xgb.XGBRegressor(**params)
          scores = cross_val_score(model, X_train, y_train, cv = 5, scoring = 'neg_root_mean_squared_error')
          return -scores.mean()
    
    class EarlyStoppingExceeded(Exception):
        pass
    
    def early_stopping_callback(study, trial):
        window_size = 20
        min_delta = 0.01
        patience = 3
    
        if not hasattr(study, '_early_stop_counter'):
            study._early_stop_counter = 0
    
        if len(study.trials) < window_size:
            return
    
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
        if len(completed_trials) < window_size:
            return
    
        best_in_window = min(t.value for t in completed_trials[-window_size:])
        global_best = study.best_value
    
        if best_in_window >= global_best - min_delta:
            study._early_stop_counter += 1
            print(f"Потенциальная остановка ({study._early_stop_counter}/{patience}): "
                  f"Лучшее в окне {best_in_window:.4f} vs глобальное {global_best:.4f}")
    
            if study._early_stop_counter >= patience:
                print(f" Вызываем остановку: нет улучшений > {min_delta} в последних {window_size*patience} trials")
                raise EarlyStoppingExceeded()
        else:
            study._early_stop_counter = 0
    
    with mlflow.start_run(run_name = 'XGBoost Optuna tuning', nested = True):
        study = optuna.create_study(direction = 'minimize', sampler = TPESampler(seed = 42))

        try:
          study.optimize(lambda trial: optimize_xgboost(trial, X_train, y_train), n_trials = 150, timeout = 600, show_progress_bar = False, callbacks = [early_stopping_callback])
        except:
          print('Ранняя остановка')
        model = xgb.XGBRegressor(**study.best_params)
        model.fit(X_train_filtered, y_train_filtered)
        y_pred = model.predict(X_test_filtered)

        cv_scores = cross_val_score(model, X_train_filtered, y_train_filtered, cv = 5, scoring = 'neg_root_mean_squared_error')

        rmse = root_mean_squared_error(y_test_filtered, y_pred)
        print(f'RMSE = {root_mean_squared_error(y_test_filtered, y_pred):.2f}')
        print(f'MAE: {mean_absolute_error(y_test_filtered, y_pred):.2f}')
        print(f'CV scores: {-1*cv_scores.mean():.2f}')

        with open('model_weights/xgb_tuned_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        signature = mlflow.models.infer_signature(X_train_filtered, y_pred)
        mlflow.log_params(study.best_params)
        mlflow.log_metric('RMSE XGBoost tuned', root_mean_squared_error(y_test_filtered, y_pred))
        mlflow.log_metric('MAE XGboost tuned', mean_absolute_error(y_test_filtered, y_pred))
        mlflow.sklearn.log_model(model, 'XGBoost tuned', signature = signature, input_example = X_train_filtered)

        print(f"Результаты показывают прогресс: {xgb_rmse:.2f} для базового градиентного бустинга и {rmse:.2f} - для тюнингованого. XGBoost после подбора гиперпараметров превзошел Random Forest.")
