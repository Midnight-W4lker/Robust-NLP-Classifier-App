# Robust NLP Classifier App

GROUP Members:

- FA23-BBD-085    MUHAMMAD ALI ABID
- FA23-0BBD-058   IBRAHIM ZAHEER
- SP23-BBD-049    Muhammad USMAN FAKHAR

## Repo Links
- https://github.com/Midnight-W4lker/Robust-NLP-Classifier-App/edit/main
- https://github.com/MAULANA-Ali-Don/Robust-NLP-Classifier-App
- https://github.com/Muhammad-Usman-09/Robust-NLP-Classifier-App/tree/main

A robust, modular, and extensible Natural Language Processing (NLP) classification pipeline for text datasets. This project supports preprocessing, model training, evaluation, and reporting, with a user-friendly Streamlit web interface.

## Features
- **🎯 Intelligent Column Auto-Detection:** 
  - Automatically detects text columns by name patterns, content length, and text characteristics
  - Auto-detects label columns by name patterns and cardinality analysis
  - Supports **both supervised** (with labels) **and unsupervised** (no labels) datasets
  - Works with diverse column names: 'text', 'review', 'content', 'sentence', 'message', 'tweet', etc.
  - Handles various label formats: 'label', 'sentiment', 'category', 'target', 'class', etc.
- **Preprocessing:** Cleans and prepares text data for modeling.
- **Multiple Models:** Supports TF-IDF, Word2Vec (CBOW), and GloVe-style (Skip-gram) representations.
- **Training & Evaluation:** Train models and evaluate on custom or provided datasets.
- **Streamlit App:** Interactive web UI for predictions and evaluation with real-time results.
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
- **Custom Datasets:** Upload any CSV file through the Streamlit app. The intelligent column detector will automatically identify text and label columns (if present).
  - **Supervised datasets** (with labels): Get full accuracy metrics and evaluation results
  - **Unsupervised datasets** (text only): Get predictions without evaluation metrics
- **Model Training:** Use `train_models.py` or `run_all.py` to train models on your data.
- **Evaluation:** Evaluate models using the Streamlit app or `predict.py`.
- **Reporting:** Generate a PDF report with `generate_report.py`.

## Data Format
The app automatically detects columns using intelligent strategies:

### Column Auto-Detection
**Text Columns** (detected by):
- Common names: `text`, `sentence`, `content`, `review`, `tweet`, `message`, `comment`, `description`, `post`, `article`, `question`, `answer`, etc.
- Average text length > 20 characters
- String datatype with longest content

**Label Columns** (detected by, *optional for unsupervised data*):
- Common names: `label`, `sentiment`, `category`, `target`, `class`, `rating`, `emotion`, `type`, etc.
- Low cardinality (2-50 unique values)
- Categorical distribution pattern

### Supported Formats
- **Supervised**: CSV with text + label columns
- **Unsupervised**: CSV with text column only (predictions without evaluation)
- **Flexible column names**: No need to rename your columns - the app adapts to your data!



## Project Demo Video

You can preview the demo video directly below (works in most Markdown viewers that support HTML5 video tags):

<p align="center">
  <video width="480" controls>
    <source src="documentation/demo.mkv" type="video/x-matroska">
    Your browser does not support the video tag.<br>
    <a href="documentation/demo.mkv">Download demo.mkv</a>
  </video>
</p>

If the video does not play in your viewer, [download the demo video](documentation/project-demo.mkv) and open it locally.

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
Project demo video on \documentation\project demo
