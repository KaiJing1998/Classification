import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', '', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r'C:\Users\KAI JING\Downloads\diabetes.csv', header=None, names=col_names)
print(data)

# determine the target and features; using features predict target
# Outcome is our target
# we use function .drop to take all other data in X, then we split the data
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', '', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
# Target variable
y = data.Outcome

# split the data into training and test data
# training data set is to train our algorithm to build a model
# testing set is to test our model to see how accurate its predictions are
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.head())
print(X_test.head())

# (154,8); test size should be 20% of the data set and the rest should be train data
# with the function of .shape you can see that we have 154 rows in the test data
print(X_test.shape)

# BUILDING DECISION TREE MODEL
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(random_state=0)

# Train Decision Tree Classifer
model = clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# EVALUATING MODEL
# Computed by comparing actual test set values and predicted values.
# model accuracy = see how often is the classifier correct ?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
from pydot import graph_from_dot_data



dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names =feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

#using ID3
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
