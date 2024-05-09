# Heart Disease Prediction

This project aims to predict the presence of heart disease based on various medical attributes of an individual. It utilizes a logistic regression model trained on a dataset containing information about patients and whether they have heart disease or not.

## Requirements

- Python 3.x
- NumPy
- Pandas
- scikit-learn

## Installation

1. Clone the repository:

```bash
git clone https://github.com/real0x0a1/heart-disease-prediction.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the dataset `data.csv` in the `content` directory.
2. Run the script `heart_disease_prediction.py`:

```bash
python heart_disease_prediction.py
```

3. The script will train a logistic regression model, evaluate its accuracy on training and test data, and make predictions for a sample input data.

## Dataset

The dataset (`data.csv`) contains the following columns:

- `age`: Age of the patient
- `sex`: Gender (0: female, 1: male)
- `cp`: Chest pain type (0-3)
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (0: false, 1: true)
- `restecg`: Resting electrocardiographic results (0-2)
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina (0: no, 1: yes)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment (0-2)
- `thal`: Thalassemia (1-3)
- `target`: Presence of heart disease (0: no, 1: yes)

## Acknowledgments

This project is inspired by the work on heart disease prediction and utilizes the dataset from the UCI Machine Learning Repository.

Feel free to customize this template according to your project specifics and preferences!

## Author

real0x0a1 

---
