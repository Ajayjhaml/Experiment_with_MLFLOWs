# import mlflow
# import mlflow.sklearn
# from sklearn.datasets import load_wine
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# import dagshub
# dagshub.init(repo_owner='Ajayjhaml', repo_name='Experiment_with_MLFLOWs', mlflow=True)

# # mlflow.set_tracking_uri("https://dagshub.com/vikashdas770/")
# mlflow.set_tracking_uri("https://dagshub.com/Ajayjhaml/Experiment_with_MLFLOWs.mlflow")

# # Load Wine dataset
# wine = load_wine()
# X = wine.data
# y = wine.target

# # Train test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# # Define the params for RF model
# max_depth = 8
# n_estimators = 5

# # Mention your experiment below
# mlflow.set_experiment('Experiment_with_MLFLOWs')

# with mlflow.start_run():
#     rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
#     rf.fit(X_train, y_train)

#     y_pred = rf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     mlflow.log_metric('accuracy', accuracy)
#     mlflow.log_param('max_depth', max_depth)
#     mlflow.log_param('n_estimators', n_estimators)

#     # Creating a confusion matrix plot
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6,6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
#     plt.ylabel('Actual')
#     plt.xlabel('Predicted')
#     plt.title('Confusion Matrix')

#     # save plot
#     plt.savefig("Confusion-matrix.png")

#     # log artifacts using mlflow
#     mlflow.log_artifact("Confusion-matrix.png")
#     mlflow.log_artifact(__file__)

#     # tags
#     mlflow.set_tags({"Author": 'Ajay', "Project": "Wine Classification"})

#     # Log the model
#     mlflow.sklearn.log_model(rf, "Random-Forest-Model")

#     print(accuracy)




import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import dagshub
import os

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner='Ajayjhaml', repo_name='Experiment_with_MLFLOWs', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Ajayjhaml/Experiment_with_MLFLOWs.mlflow")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Parameters for Random Forest
max_depth = 8
n_estimators = 5

# Set experiment name
mlflow.set_experiment('Experiment_with_MLFLOWs')

with mlflow.start_run():
    # Train model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save and log confusion matrix
    cm_file = "Confusion-matrix.png"
    plt.savefig(cm_file)
    mlflow.log_artifact(cm_file)

    # Save and log model manually instead of log_model (unsupported by DagsHub)
    model_path = "random_forest_model.pkl"
    joblib.dump(rf, model_path)
    mlflow.log_artifact(model_path)

    # Log this script file (optional)
    current_script = os.path.abspath(__file__)
    mlflow.log_artifact(current_script)

    # Add tags
    mlflow.set_tags({"Author": "Ajay", "Project": "Wine Classification"})

    print(f"✅ Accuracy: {accuracy:.4f}")
