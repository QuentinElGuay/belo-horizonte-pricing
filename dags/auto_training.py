from datetime import datetime, timedelta
from airflow import DAG
from airflow.models.param import Param
from airflow.operators.docker_operator import DockerOperator
from airflow.operators.dummy_operator import DummyOperator

default_args = {
'owner'                 : 'airflow',
'description'           : 'Use of the DockerOperator',
'depend_on_past'        : False,
'start_date'            : datetime(2021, 5, 1),
'email_on_failure'      : False,
'email_on_retry'        : False,
'retries'               : 1,
'retry_delay'           : timedelta(minutes=5)
}

with DAG(
    'docker_operator_dag',
    default_args=default_args,
    schedule_interval='* * 1 * *',
    catchup=False,
    params={
        'experience_name': Param(
            '', type='string', description='The name of the experience to run.'),
        'dataset_uri': Param(
            '',
            type='string',
            description='The path to the dataset to use.',
            format='uri-template'
        ),
    }
) as dag:

    start_dag = DummyOperator(
        task_id='start_dag'
        )

    end_dag = DummyOperator(
        task_id='end_dag'
        )        

    t1 = DockerOperator(
        task_id='docker_prepare_dataset',
        api_version='auto',
        container_name='task___prepare_dataset',
        image='mlops/mlops:0.1.0',
        auto_remove=True,
        command='prepare-dataset {{ params.experience_name }} {{ params.dataset_uri }}',
        docker_url='unix://var/run/docker.sock',
        network_mode='belo-horizonte-pricing_backend',
        environment={
            'AWS_ACCESS_KEY_ID': 'XeAMQQjZY2pTcXWfxh4H',
            'AWS_SECRET_ACCESS_KEY': 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ',
            'S3_ENDPOINT_URL': 'http://s3:9000/',  #TODO: check the official ENV var again
            'DATASET_BUCKET': 'mlops-datasets'
        }
    )

    t2 = DockerOperator(
        task_id='docker_train_model_simple_linear_regression',
        api_version='auto',
        container_name='task___train_model_simple_linear_regression',
        image='mlops/mlops:0.1.0',
        auto_remove=True,
        command='train-model {{ params.experience_name }} simple_linear_regression',
        docker_url='unix://var/run/docker.sock',
        network_mode='belo-horizonte-pricing_backend',
        environment={
            'AWS_ACCESS_KEY_ID': 'XeAMQQjZY2pTcXWfxh4H',
            'AWS_SECRET_ACCESS_KEY': 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ',
            'S3_ENDPOINT_URL': 'http://s3:9000/',
            'TRACKING_SERVER_URI': 'http://mlflow:5000/',
            'DATASET_BUCKET': 'mlops-datasets'
        }
    )

    t3 = DockerOperator(
        task_id='docker_train_model_elasticnet',
        api_version='auto',
        container_name='task___train_model_elasticnet',
        image='mlops/mlops:0.1.0',
        auto_remove=True,
        command='train-model {{ params.experience_name }} elastic_net_regression',
        docker_url='unix://var/run/docker.sock',
        network_mode='belo-horizonte-pricing_backend',
        environment={
            'AWS_ACCESS_KEY_ID': 'XeAMQQjZY2pTcXWfxh4H',
            'AWS_SECRET_ACCESS_KEY': 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ',
            'S3_ENDPOINT_URL': 'http://s3:9000/',
            'TRACKING_SERVER_URI': 'http://mlflow:5000/',
            'DATASET_BUCKET': 'mlops-datasets'
        }
    )

    t4 = DockerOperator(
        task_id='docker_evaluate_experience',
        api_version='auto',
        container_name='task___evaluate_experience',
        image='mlops/mlops:0.1.0',
        auto_remove=True,
        command='evaluate-experience {{ params.experience_name }}',
        docker_url='unix://var/run/docker.sock',
        network_mode='belo-horizonte-pricing_backend',
        environment={
            'AWS_ACCESS_KEY_ID': 'XeAMQQjZY2pTcXWfxh4H',
            'AWS_SECRET_ACCESS_KEY': 'wyJ30G38aC2UcyaFjVj2dmXs1bITYkJBcx0FtljZ',
            'S3_ENDPOINT_URL': 'http://s3:9000/',
            'TRACKING_SERVER_URI': 'http://mlflow:5000/',
            'DATASET_BUCKET': 'mlops-datasets'
        }
    )

    start_dag >> t1 >> [t2, t3] >> t4 >> end_dag
