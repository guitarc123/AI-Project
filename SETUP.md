# Setup Instructions

## Generate Model and Confusion Matrix

Follow these steps to generate the trained Iris model and confusion matrix visualization:

### 1. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Training Script

```bash
python src/train.py
```

This will:
- Train a RandomForest classifier on the Iris dataset
- Save the trained model to `outputs/iris_model.joblib`
- Generate and save the confusion matrix to `outputs/confusion_matrix.png`
- Print accuracy and classification metrics

### 4. Output Files

After running the training script, you'll find:
- **Model**: `outputs/iris_model.joblib` (serialized scikit-learn model)
- **Confusion Matrix**: `outputs/confusion_matrix.png` (visualization)

## Project Structure

```
AI-Project/
├── src/
│   └── train.py              # Main training script
├── notebooks/
│   └── iris_classifier.ipynb # Jupyter notebook for EDA
├── outputs/                  # Generated files (gitignored)
│   ├── iris_model.joblib
│   └── confusion_matrix.png
├── requirements.txt          # Project dependencies
├── README.md                # Project overview
└── SETUP.md                 # This file
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
- Solution: Make sure you've installed requirements: `pip install -r requirements.txt`

**Issue**: `outputs` directory doesn't exist
- Solution: The script creates it automatically. If issues persist, create manually: `mkdir outputs`

## Next Steps

After generating the artifacts:
1. Check `outputs/confusion_matrix.png` to visualize model performance
2. Use `outputs/iris_model.joblib` to make predictions on new data
3. Review `src/train.py` for model configuration and training parameters
