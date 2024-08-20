IMAGE_NAME=mlops/mlops
IMAGE_TAG=0.1.0

tests:
	pytest tests/

quality_checks:
	isort .
	blue .

build: quality_checks tests
	docker build -t ${IMAGE_NAME}:${IMAGE_TAG} src/.

init_airflow: build
	docker compose up airflow-init
