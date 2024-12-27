from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("DATA-30-AD.csv")

X = data.drop(columns=['Success'])
Y = data['Success']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

classifiers = {
    "Random Forest": RandomForestClassifier(),
    "SVC": SVC(kernel='linear', C=3),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=3),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Multi-layer Perceptron": MLPClassifier(),
    "SVM with RBF Kernel": SVC(kernel='rbf')
}



for name, clf in classifiers.items():

        clf.fit(X_train, Y_train)
        accuracy = clf.score(X_test, Y_test) * 100

        print(f"Test accuracy with {name}: {accuracy:.2f}%")

