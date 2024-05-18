import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv('student_stress_factors.csv')
dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=True)
# print(dt_Train)

X_train = dt_Train.iloc[:, :5]
y_train = dt_Train.iloc[:, 5]
X_test = dt_Test.iloc[:, :5]
y_test = dt_Test.iloc[:, 5]

# print(X_train)
# print(y_train)

# Perceptron
perceptron_model = Perceptron(penalty='l1', class_weight='balanced')
perceptron_model.fit(X_train, y_train)
y_pred_perceptron = perceptron_model.predict(X_test)

# SVM
svm_model = SVC(kernel='poly', degree=10)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Neural Network
nn_model = MLPClassifier(solver='lbfgs', max_iter=3000)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)

# Logistic Regression
logistic_model = LogisticRegression(solver='liblinear', max_iter=3000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"{model_name}:")
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n')

evaluate_model(y_test, y_pred_perceptron, "Perceptron")
evaluate_model(y_test, y_pred_svm, "SVM")
evaluate_model(y_test, y_pred_tree, "Decision Tree")
evaluate_model(y_test, y_pred_nn, "Neural Network")
evaluate_model(y_test, y_pred_logistic, "Logistic Regression")

# Code cho phần vẽ biểu đồ
def plot_evaluation_metrics(models, metrics, scores, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.2  # Kích thước của mỗi cột
    x = range(len(models))

    for i, metric in enumerate(metrics):
        ax.bar([pos + width * i for pos in x], scores[:, i], width=width, label=metric)

    ax.set_xticks([pos + width * (len(metrics) - 1) / 2 for pos in x])
    ax.set_xticklabels(models)
    ax.set_xlabel('Models')
    ax.set_title(title)
    ax.legend()

    plt.show()

# Đánh giá các mô hình
models = ["Perceptron", "SVM", "Decision Tree", "Neural Network", "Logistic Regression"]
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# Lưu trữ kết quả đánh giá của từng mô hình và từng độ đo
scores = []
for model in [perceptron_model, svm_model, tree_model, nn_model, logistic_model]:
    y_pred = model.predict(X_test)
    scores.append([accuracy_score(y_test, y_pred),
                   precision_score(y_test, y_pred, average='macro'),
                   recall_score(y_test, y_pred, average='macro'),
                   f1_score(y_test, y_pred, average='macro')])

scores = np.array(scores)

# Vẽ biểu đồ
plot_evaluation_metrics(models, metrics, scores, "Model Evaluation Metrics")

