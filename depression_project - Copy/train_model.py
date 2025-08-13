import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

data = np.load("dataset/features_labels.npz")
X = data["X"]
y = data["y"]

# ReliefF feature selection
fs = ReliefF(n_features_to_select=30)
X_sel = fs.fit_transform(X, y)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sel)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=["No", "Mild", "Moderate", "Severe"]))

os.makedirs("models", exist_ok=True)
dump(clf, "models/classifier.joblib")
dump(scaler, "models/scaler.joblib")
np.save("models/top_feature_indices.npy", fs.top_features_[:30])

print("✅ Model and scaler saved in models/")
