import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE


csv= r"\\spatialfiles.bcgov\work\ilmb\dss\dss_workarea\mlabiadh\workspace\20241118_land_classification\data\TrainingFeatureCollectionCSV.csv"

df= pd.read_csv(csv)


X = df.drop(columns=['class_id'])
y = df['class_id']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)


scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")


y_pred = model.predict(X_test)
confusion_matrix= confusion_matrix(y_test, y_pred) 
classification_report= classification_report(y_test, y_pred)


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=10)
plt.colorbar()
plt.show()