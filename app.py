from flask import Flask, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

app = Flask(__name__)
mlflow.set_tracking_uri("http://host.docker.internal:5050")
client = MlflowClient()

try:
    experiment = client.get_experiment_by_name('Fireforest')
    if experiment is None:
        raise Exception("Experiment not found")
    runs = client.search_runs(experiment.experiment_id, order_by = ["start_time DESC"], max_results=1)
    if not runs:
        raise Exception("Run not found")
    RUN_ID = runs[0].info.run_id
except Exception as e:
    print(e)

model_path = f"runs:/{RUN_ID}/XGBoost tuned"
model = mlflow.xgboost.load_model(model_path)

FEATURES = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
       'month_apr', 'month_aug', 'month_dec', 'month_feb', 'month_jan',
       'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov',
       'month_oct', 'month_sep', 'day_fri', 'day_mon', 'day_sat', 'day_sun',
       'day_thu', 'day_tue', 'day_wed']

@app.route('/', methods = ['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Forestfires Prediction Model'})

@app.route('/info', methods = ['GET'])
def info():
    try:
        run = client.get_run(RUN_ID)
        metric_dict  = {k: v for k, v in run.data.metrics.items()}
        params = run.data.params
        param_dict = {k: v for k, v in params.items()}
        return jsonify({'status': 'ok', 'name': 'XGBoost tuned', 'metrics': metric_dict, 'params': param_dict})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'})
        features = np.array([data[feature] for feature in FEATURES]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return jsonify({'status': 'ok', 'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)





