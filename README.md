# Robust NLP Classifier App

A robust, modular, and extensible Natural Language Processing (NLP) classification pipeline for text datasets. This project supports preprocessing, model training, evaluation, and reporting, with a user-friendly Streamlit web interface.

## Features
- **Automatic Text & Label Column Detection:** Handles various dataset formats by auto-detecting text and label columns.
- **Preprocessing:** Cleans and prepares text data for modeling.
- **Multiple Models:** Supports Word2Vec, GloVe, and other models for text representation.
- **Training & Evaluation:** Train models and evaluate on custom or provided datasets.
- **Streamlit App:** Interactive web UI for predictions and evaluation.
- **Report Generation:** Automated PDF report summarizing results.

## Project Structure
```
├── api.py                  # API endpoints for model inference
├── generate_datasets.py    # Scripts to generate or preprocess datasets
├── generate_report.py      # Generates PDF reports from results
├── predict.py              # Prediction and evaluation logic
├── preprocessing.py        # Text preprocessing utilities
├── requirements.txt        # Python dependencies
├── run_all.py              # Runs the full pipeline (preprocessing, training, evaluation, report)
├── streamlit_app.py        # Streamlit web application
├── train_models.py         # Model training scripts
├── data/                   # Datasets (train/test CSVs, label maps)
├── documentation/          # Additional documentation
├── models/                 # Saved models and training results
```

## Quick Start
1. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Run the full pipeline:**
	```bash
	python run_all.py
	```
3. **Launch the Streamlit app:**
	```bash
	streamlit run streamlit_app.py
	```
4. **Generate a report:**
	```bash
	python generate_report.py
	```

## Usage
- **Custom Datasets:** Place your CSV files in the `data/` folder. The app will auto-detect text and label columns.
- **Model Training:** Use `train_models.py` or `run_all.py` to train models on your data.
- **Evaluation:** Evaluate models using the Streamlit app or `predict.py`.
- **Reporting:** Generate a PDF report with `generate_report.py`.

## Data Format
- CSV files should contain at least one text column (e.g., `text`, `sentence`, `review`) and one label column (e.g., `label`, `target`, `class`).
- The pipeline auto-detects these columns.


## Project Demo Video


Watch the demo video to see the app in action:

<p align="center">
	<a href="https://www.youtube.com/watch?v=YOUR_DEMO_VIDEO_LINK" target="_blank">
		<img src="https://img.youtube.com/vi/YOUR_DEMO_VIDEO_LINK/0.jpg" alt="Project Demo Video" width="480"/>
	</a>
</p>

Or watch directly below:

<div align="center">
	<iframe width="560" height="315" src="https://www.youtube.com/embed/YOUR_DEMO_VIDEO_LINK" title="Project Demo Video" frameborder="0" allowfullscreen></iframe>
</div>

## Extending the Project
- Add new models in `train_models.py` and update `predict.py` for inference.
- Add new preprocessing steps in `preprocessing.py`.
- Update the Streamlit UI in `streamlit_app.py` as needed.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
MIT License

## Authors
- [Midnight-W4lker](https://github.com/Midnight-W4lker)

---
For more details, see the code comments and documentation folder.
# Robust-NLP-Classifier-App
