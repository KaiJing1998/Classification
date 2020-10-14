import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from io import StringIO

# import data
row_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r'C:\Users\KAI JING\Downloads\diabetes.csv', header=0, names=row_names)
print(data)

# feature selection
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
X = data.drop("Outcome", axis=1)
# Target variable
y = data.Outcome

# slit the training and testing data randomly into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Training Data 80% :", X_train.shape)
print("Testing Data 20% : ", X_test.shape)

"""TRAINING DATA SET DECISION TREE; MAKE PREDICTIONS WITH TESTING DATA"""
# using training data set to create a decision tree
clf = DecisionTreeClassifier(criterion="entropy")
model = clf.fit(X_train, y_train)
# use the model to make predictions with the testing data
y_pred = clf.predict(X_test)
# Difference between the actual value and predicted value.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
# plot the comparision of Actual and Predicted values
df.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
# how did our model perform ?
count_misclassified = (y_test != y_pred).sum()
print('Training data set predict with the testing data set')
print('Misclassified samples: {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}'.format(accuracy))

# visualization of the decision graph
# export_graphviz function converts decision tree classifier into dot file
# and pydotplus convert this dot file to png
# then it will display it at your project files
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('training.png')
Image(graph.create_png())

"""TESTING DATA SET DECISION TREE"""
# using testing data set to create a decision tree
clf = DecisionTreeClassifier(criterion="entropy")
model = clf.fit(X_test, y_test)
# visualization of the decision graph
# export_graphviz function converts decision tree classifier into dot file
# and pydotplus convert this dot file to png
# then it will display it at your project files
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=feature_cols, class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('testing.png')
Image(graph.create_png())
