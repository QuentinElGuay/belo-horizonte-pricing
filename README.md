# belo-horizonte-pricing
## Introduction
This project was developped in the context of the [DataTalks.Club MLOps Zoomcamp course](#mlops-zoomcamp-project) and serves as a demonstration project for my knowledge in the implementation of `MLOps methodologies`.

### What is MLOps
According to [Databricks](https://www.databricks.com/glossary/mlops), _"MLOps stands for `Machine Learning Operations` [and focus] on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them"_. In other words, it is a combination of `cultural philosophies`, `good practices` and `tools` inspired from the [DevOps methodologies](https://aws.amazon.com/devops/what-is-devops/) for software and applied to the specif necessities of `Machine Learning` with the intention of increasing an organization’s ability to deliver, then evolve and improve, Machine Learning models at `high velocity` and in a `leaner` way.

### The MLOps process
The process we want to implement in this project is the following:
![The illustrated flow of an MLOps project](docs/pictures/mlops.jpg "MLOps Flow")

1. Experimentation phase:
    1. Comes a first dataset, analyzed and cleand by a data scientist.
    2. The data scientist runs multiples experiments to find the best possible model:
        1. Split the dataset in two to keep a test dataset used to decide between the challengers.
        2. Run experiments on the training set (algorithm and optimization) using the `MLFlow Tracking Server` to store the results.
        3. Select the best candidates and apply them to the test dataset.
        4. Register the best candidate in the `MLFlow Model Registery` as current champion (model tu use in production).
2. Deployment in production:
    1. New values are sent for prediction, either via API or a Batch process.
    2. The code calls the model register to load the current production model.
    3. Results are returned to the user and stored for monitoring.
3. Feedback loop:
    1. On scheduled dates or when a datashift is detected, create a new dataset using recente predictions and actual values.
    2. Execute the model training process automatically using the new dataset (see 1.).
    3. Compare the result of the current model in production with the new champion.
    4. Promote (automatically or not) the new champion if required.

### About this particular project
I decided to use the [**house-pricing-in-belo-horizonte** dataset](https://www.kaggle.com/datasets/guilherme26/house-pricing-in-belo-horizonte) available on Kaggle to try to solve the classic `house pricing prediction` problem. This dataset is _rather small_ and somewhat _limitated in the quantity of features_ as demonstrated by the [Explory Data Analysis](EDA.ipynb). This implies that our capacity to predict precisely the prices will be limited. It is however not really a problem since the main goal of this project is to demonstrate the MLOps methodologies rather than pure Machine Learning technics.

## Install the project
TODO

## Run the project
TODO

## MLOps Zoomcamp Project
This project was developped for the [DataTalks.Club MLOps Zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp).

### Self-Evaluation
**Problem description:**
- [ ] 0 points: The problem is not described
- [x] 1 point: The problem is described but shortly or not clearly
- [ ] 2 points: The problem is well described and it's clear what the problem the project solves

**Cloud:**
- [ ] 0 points: Cloud is not used, things run only locally
- [ ] 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
- [x] 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure:
    - IaC: project using the `AWS SAM framework` to manage an `AWS CloudFormation stack` (template.yaml).
    - Creation of an `API Gateway endpoint` calling an `AWS Lambda function` to return the prediction value.

**Experiment tracking and model registry:**
- [ ] 0 points: No experiment tracking or model registry
- [ ] 2 points: Experiments are tracked or models are registered in the registry
- [x] 4 points: Both experiment tracking and model registry are used

**Workflow orchestration:**
- [x] 0 points: No workflow orchestration
- [ ] 2 points: Basic workflow orchestration
- [ ] 4 points: Fully deployed workflow

**Model deployment:**
- [ ] 0 points: Model is not deployed
- [ ] 2 points: Model is deployed but only locally
- [x] 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used

**Model monitoring:**
- [x] 0 points: No model monitoring
- [ ] 2 points: Basic model monitoring that calculates and reports metrics
- [ ] 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated

**Reproducibility:**
- [x] 0 points: No instructions on how to run the code at all, the data is missing
- [ ] 2 points: Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing
- [ ] 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.

**Best practices:**
- [x] There are unit tests (1 point):
    - The coverage isn't yet complete but you can run `pytest` to execute some unit tests.
- [ ] There is an integration test (1 point)
- [x] Linter and/or code formatter are used (1 point)
    - Execute `poetry run blue .` to use the [blue code formatter](https://pypi.org/project/blue/).
    - Execute `poetry run isort .` to use the `isort`.
- [ ] There's a Makefile (1 point)
- [ ] There are pre-commit hooks (1 point)
- [ ] There's a CI/CD pipeline (2 points)
