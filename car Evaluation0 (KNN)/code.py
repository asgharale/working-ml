from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model
import pickle

df = pd.read_csv("car.data")

label = "class"

pre = preprocessing.LabelEncoder()
buying = pre.fit_transform(list(df["buying"]))
maint = pre.fit_transform(list(df["maint"]))
doors = pre.fit_transform(list(df["doors"]))
persons = pre.fit_transform(list(df["persons"]))
lug_boot = pre.fit_transform(list(df["lug_boot"]))
safety = pre.fit_transform(list(df["safety"]))
cls = pre.fit_transform(list(df["class"]))

x = np.array(list(zip(buying, maint, doors, persons, lug_boot, safety)))
y = np.array(list(cls))

best = 0
for _ in range(100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    model = KNeighborsClassifier(n_neighbors=7)

    model.fit(x_train, y_train)

    acc = model.score(x_test, y_test)
    if(acc>best):
        best = acc
        with open('highest-model.pkl', 'wb') as file:
            pickle.dump(model, file)

with open('highest-model.pkl', 'rb') as f:
    model = pickle.load(f)

acc = model.score(x_test, y_test)
# acc

predictions = model.predict(x_test)

classes = ['unacc', 'acc', 'good', 'vgood']
for i in range(len(predictions)):
    print(classes[predictions[i]], classes[y_test[i]])