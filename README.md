# Iris Classifier Project

A beginner-friendly machine learning project using the Iris dataset to build a classification model.

## Project Structure

```
ml-project/
├── notebooks/
│   └── iris_classifier.ipynb
├── src/
│   └── train.py
├── data/
├── outputs/
│   ├── confusion_matrix.png
│   └── iris_model.joblib
├── venv/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the training script:**
   ```bash
   python src/train.py
   ```

## Project Workflow

1. **Jupyter Notebook** (`notebooks/iris_classifier.ipynb`): Exploratory data analysis and model development
2. **Training Script** (`src/train.py`): Standalone Python script that trains the model and saves outputs
3. **Outputs** (`outputs/`): Contains the trained model and confusion matrix visualization

## Technologies Used

- **scikit-learn**: Machine learning library
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter**: Interactive notebook environment
- **joblib**: Model serialization

## Files

- `train.py`: Main training script that builds the Iris classifier, saves the model, and generates a confusion matrix plot
- `iris_classifier.ipynb`: Jupyter notebook with exploratory analysis and model development
- `requirements.txt`: Project dependencies
