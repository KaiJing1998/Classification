import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from io import StringIO

# import data
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r'C:\Users\KAI JING\Downloads\diabetes.csv', header=0, names=col_names)
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
print("Testing Data 20% : ", X_test.shape)

# using training data set to create a decision tree
clf = DecisionTreeClassifier(criterion="entropy")
model = clf.fit(X_train, y_train)
# use the model to make predictions with the testing data
y_pred = clf.predict(X_test)
# how did our model perform ?
count_misclassified = (y_test != y_pred).sum()
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
graph.write_png('Diabetes.png')
Image(graph.create_png())




