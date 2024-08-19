import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
df = pd.read_csv("music.csv")
x, y = df.drop(columns=['genre']), df['genre']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# dumping my model into joblib file
joblib.dump(model, 'music-model.joblib')
test = joblib.load("music-model.joblib")

# model = test
predictions = test.predict(x_test)
# predictions

score = accuracy_score(y_test, predictions)
score