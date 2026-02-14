from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump

# Create dummy training data with 30 features (same as your feature.py)
X_train, y_train = make_classification(
    n_samples=200,
    n_features=30,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Train the model
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)

# Save the new model
dump(gbc, "model/model_compatible.joblib")
print("✅ Model trained and saved successfully as model_compatible.joblib")
