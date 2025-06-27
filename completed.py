import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # ensure headless mode (no GUI required)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
DATA_PATH = 'essay_llama3_8B_groq.csv'
MODEL_PATH = 'model.joblib'
VECT_PATH = 'vectorizer.joblib'
OUTPUT_DIR = 'docs'
REPORT_PATH = os.path.join(OUTPUT_DIR, 'report.md')

# === SETUP ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA AND MODEL ===
print('Loading data and model...')
df = pd.read_csv(DATA_PATH)
essays = df['cleaned_text'].fillna('') if 'cleaned_text' in df.columns else df['full_text'].fillna('')
scores = df['score'].astype(float)

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)
print('Model and vectorizer loaded.')

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    essays, scores, test_size=0.2, random_state=42
)
print(f'Testing on {len(X_test)} essays.')

# === PREDICT & EVALUATE ===
print('Vectorizing test essays and predicting...')
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = (np.rint(y_pred) == y_test).mean() * 100
print('Evaluation complete.')

metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'ExactMatch%': accuracy}

# === PLOTS ===
print('Generating plots...')

# True vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel('True Score')
plt.ylabel('Predicted Score')
plt.title('True vs. Predicted Essay Scores')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'true_vs_predicted.png'))
plt.close()

# Residuals
plt.figure(figsize=(6, 4))
residuals = y_test - y_pred
plt.hist(residuals, bins=30, color='gray', edgecolor='black')
plt.xlabel('Residual (True - Predicted)')
plt.ylabel('Frequency')
plt.title('Residuals Histogram')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_histogram.png'))
plt.close()

# Feature Importances
importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
if importances is not None:
    indices = np.argsort(importances)[-20:]
    fnames = np.array(vectorizer.get_feature_names_out())[indices]
    plt.figure(figsize=(10, 6))
    plt.barh(fnames, importances[indices], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 20 TF-IDF Feature Importances')
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importances.png'))
    plt.close()

print('Plots saved to docs/.')

# === REPORT ===
print('Writing report...')
with open(REPORT_PATH, 'w') as f:
    f.write('# Project Evaluation Report\n\n')
    f.write('## 1. Technical Implementation\n')
    f.write('- Used TF-IDF vectorizer and RandomForestRegressor from scikit-learn.\n\n')
    f.write('## 2. Model Performance\n')
    for k, v in metrics.items():
        f.write(f'- **{k}**: {v:.4f}\n')
    f.write('\n**Plots:**\n')
    f.write(f'![True vs Predicted](true_vs_predicted.png)\n')
    f.write(f'![Residuals Histogram](residuals_histogram.png)\n')
    if importances is not None:
        f.write(f'![Feature Importances](feature_importances.png)\n')
    f.write('\n')
    f.write('## 3. Documentation and Reporting\n')
    f.write('- This report details model design, data prep, evaluation metrics, and visuals.\n\n')
    f.write('## 4. Practical Relevance\n')
    f.write('- Streamlit UI available via `app.py` to grade essays interactively.\n')
    f.write('- Limitations: short essays (<50 words) may produce unstable scores.\n')
    f.write('- Future work: add retraining, explainability, and handle data drift.\n')

print(f'Report successfully written to: {REPORT_PATH}')
