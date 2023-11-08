import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

md = pd.read_csv("music.csv")
x = md.drop(columns=["genre"])
y = md["genre"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))