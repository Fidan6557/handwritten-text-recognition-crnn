.PHONY: install test train demo clean lint plot-metrics predict-line predict-page prepare-iam prepare-hf-iam

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

train:
	python -m src.training.train --config configs/config.yaml

demo:
	streamlit run app/streamlit_app.py

prepare-iam:
	python -m src.data.prepare_iam

prepare-hf-iam:
	python -m src.data.prepare_hf_iam

predict-line:
	@if [ -z "$(IMAGE)" ]; then echo "Usage: make predict-line IMAGE=path/to/line.png"; exit 1; fi
	python -m src.inference.predict --image "$(IMAGE)"

predict-page:
	python -m src.inference.predict_page --image "$(or $(IMAGE),sample_images/page_sample.jpg)"

plot-metrics:
	python -m src.visualization.plot_metrics

lint:
	ruff check src/ tests/ app/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
