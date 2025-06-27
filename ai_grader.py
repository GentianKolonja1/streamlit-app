import joblib # for saving and loading models
import json # for loading data
import pandas as pd # For visualizations
import numpy as np # for numerical computations
from sklearn.ensemble import RandomForestRegressor # For regression
from sklearn.feature_extraction.text import TfidfVectorizer # For text data
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.metrics import(  # For evaluation metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt

# 1) Loading dataset
df = pd.read_csv("essay_llama3_8B_groq.csv")

# 2) # Preprocessing
print("Dataset columns:", df.columns.tolist())
print(df[['cleaned_text', 'score']].head(), "\n")

# 3) Features & target
X = df['cleaned_text'].fillna("")
y = df['score']

# 4) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training on {len(X_train)} essays; testing on {len(X_test)} essays.\n")

 # 5) Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)
X_tr = vectorizer.fit_transform(X_train)
X_te = vectorizer.transform(X_test)
print(" TF-IDF vectorization done.\n")

# 6) Train RF regressor faster
print("⏳ Training model .. May take some time")
model = RandomForestRegressor(
    n_estimators=50,    # fewer trees for speed
    n_jobs=-1,          # use all CPU cores
    random_state=42
)
model.fit(X_tr, y_train)
print(" Model training complete.\n")

# 7) Save model & vectorizer
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(model,      "model.joblib")
print(" Artifacts saved (vectorizer.joblib, model.joblib)\n")

# 8) Predict & evaluate
y_pred = model.predict(X_te)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
r2_pct = r2 * 100

# exact‐match accuracy
y_pred_round = np.rint(y_pred).astype(int)
y_pred_round = np.clip(y_pred_round, int(y_test.min()), int(y_test.max()))
acc_pct = (y_pred_round == y_test.values).mean() * 100

# 9) Print metrics
print("Evaluation Metrics:")
print(f"  • MSE  : {mse:.3f}")
print(f"  • RMSE : {rmse:.3f}")
print(f"  • MAE  : {mae:.3f}")
print(f"  • R²   : {r2:.3f} ({r2_pct:.1f}% variance explained)")
print(f"  • Accuracy (exact‐match): {acc_pct:.1f}%\n")

# 10) Save metrics for app
metrics = {
    "mse":  mse,
    "rmse": rmse,
    "mae":  mae,
    "r2":   r2,
    "variance_pct": r2_pct,
    "accuracy_pct": acc_pct
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
print("✅ Saved metrics.json for app display.\n")

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k')
plt.xlabel('True Scores')
plt.ylabel('Predicted Scores')
plt.title('True vs. Predicted Essay Scores')
plt.tight_layout()
plt.show()
