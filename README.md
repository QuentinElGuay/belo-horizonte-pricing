# belo-horizonte-pricing
## Introduction
This project was developped in the context of the [DataTalks.Club MLOps Zoomcamp course](#mlops-zoomcamp-project) and serves as a demonstration project for my knowledge in the implementation of `MLOps methodologies`.

### What is MLOps
According to [Databricks](https://www.databricks.com/glossary/mlops), _"MLOps stands for `Machine Learning Operations` [and focus] on streamlining the process of taking machine learning models to production, and then maintaining and monitoring them"_. In other words, it is a combination of `cultural philosophies`, `good practices` and `tools` inspired from the [DevOps methodologies](https://aws.amazon.com/devops/what-is-devops/) for software and applied to the specif necessities of `Machine Learning` with the intention of increasing an organization’s ability to deliver, then evolve and improve, Machine Learning models at `high velocity` and in a `leaner` way.

### About this particular project
I decided to use the [**house-pricing-in-belo-horizonte** dataset](https://www.kaggle.com/datasets/guilherme26/house-pricing-in-belo-horizonte) available on Kaggle to try to solve the classic `house pricing prediction` problem. This dataset is _rather small_ and somewhat _limitated in the quantity of features_ as demonstrated by the [Explory Data Analysis](EDA.ipynb). This implies that our capacity to predict precisely the prices will be limited. It is however not really a problem since the main goal of this project is to demonstrate the MLOps methodologies rather than pure Machine Learning technics.

## Install the project
TODO

## Run the project
TODO

## MLOps Zoomcamp Project
This project was developped for the [DataTalks.Club MLOps Zoomcamp course](https://github.com/DataTalksClub/mlops-zoomcamp).

### Evaluation Criteria
**Problem description:**
- [ ] 0 points: The problem is not described
- [ ] 1 point: The problem is described but shortly or not clearly
- [ ] 2 points: The problem is well described and it's clear what the problem the project solves

**Cloud:**
- [ ] 0 points: Cloud is not used, things run only locally
- [ ] 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
- [ ] 4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure

**Experiment tracking and model registry:**
- [ ] 0 points: No experiment tracking or model registry
- [ ] 2 points: Experiments are tracked or models are registered in the registry
- [ ] 4 points: Both experiment tracking and model registry are used

**Workflow orchestration:**
- [ ] 0 points: No workflow orchestration
- [ ] 2 points: Basic workflow orchestration
- [ ] 4 points: Fully deployed workflow

**Model deployment:**
- [ ] 0 points: Model is not deployed
- [ ] 2 points: Model is deployed but only locally
- [ ] 4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used

**Model monitoring:**
- [ ] 0 points: No model monitoring
- [ ] 2 points: Basic model monitoring that calculates and reports metrics
- [ ] 4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated

**Reproducibility:**
- [ ] 0 points: No instructions on how to run the code at all, the data is missing
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
- [ ] There are pre- [ ]commit hooks (1 point)
- [ ] There's a CI/CD pipeline (2 points)
