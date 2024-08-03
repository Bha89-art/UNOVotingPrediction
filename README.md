import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import joblib
# Load the dataset
data = pd.read_csv('un_voting_data0.csv')

# Check columns in the dataset
print("Columns in dataset:", data.columns)
# Data Cleaning
vote_mapping = {'abstained': 0, 'no': -1, 'yes': 1}
for country in ['China', 'France', 'Russia', 'UK', 'USA']:
    if country in data.columns:
        data[country] = data[country].map(vote_mapping)
    else:
        print(f"Warning: Column {country} not found in the dataset.")
# Identify numeric and categorical columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
categorical_cols = data.select_dtypes(include=[object]).columns

# Impute missing values
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

print("Number of NaN values after processing:", data.isna().sum().sum())
# Feature Engineering
if all(col in data.columns for col in ['China', 'France', 'Russia', 'UK']):
    data['Average_Vote'] = data[['China', 'France', 'Russia', 'UK']].mean(axis=1)

# Time-Series Analysis (assuming 'Year' is in the dataset)
if 'Year' in data.columns:
    if data['Year'].dtype != 'int64':
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

    time_series_data = data.groupby('Year').mean(numeric_only=True).reset_index()
else:
    print("Warning: 'Year' column not found in the dataset.")
    time_series_data = pd.DataFrame()  # Empty DataFrame for visualization
# Clustering and PCA for feature engineering
if 'USA' in data.columns:
    X = data.drop(['USA', 'UNO Topics'], axis=1, errors='ignore')
    y = data['USA']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    # Train and evaluate models
for model_name, model in models.items():
        pipeline = Pipeline([
('scaler', StandardScaler()),
('model', model)])
       
        # Train the model
        pipeline.fit(X_train, y_train)
       
        # Predictions
        y_pred = pipeline.predict(X_test)
       
        # Evaluation
        print(f"\nModel: {model_name}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(vote_mapping.keys()), yticklabels=sorted(vote_mapping.keys()))
        plt.title(f'Confusion Matrix for {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        # Feature Importance (for models that support it)
if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure()
            plt.title(f"Feature Importances for {model_name}")
            plt.bar(range(X.shape[1]), importances[indices], align="center")
            plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
            plt.xlim([-1, X.shape[1]])
            plt.show()
            # Clustering and PCA for feature engineering
pca = PCA(n_components=2)
X = data.drop(['UNO Topics', 'Year'], axis=1, errors='ignore')  # Exclude non-numeric columns for PCA
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('PCA of Feature Data with Clusters')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
# Clustering and PCA for feature engineering
pca = PCA(n_components=2)
X = data.drop(['UNO Topics', 'Year'], axis=1, errors='ignore')  # Exclude non-numeric columns for PCA
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X)
# Interactive Visualization with Plotly
if not time_series_data.empty:
    fig = px.line(time_series_data, x='Year', y=['China', 'France', 'Russia', 'UK'], title='Voting Patterns Over Time')
    fig.update_layout(xaxis_title='Year', yaxis_title='Average Vote')
    fig.show()
        
