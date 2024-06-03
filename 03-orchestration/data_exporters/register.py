import mlflow
import pickle

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("hw3_experiment")

@data_exporter
def export_data(data, *args, **kwargs):

    dv, lr = data

    with open('/home/src/mlops/homework_03/custom/dict_vectorizer', 'wb') as f_out:
        pickle.dump(dv, f_out)  

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, artifact_path="models")
        mlflow.log_artifact('/home/src/mlops/homework_03/custom/dict_vectorizer')


