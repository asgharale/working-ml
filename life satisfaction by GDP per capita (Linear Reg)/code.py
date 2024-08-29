import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("lifesat_full.csv")

x = df[["GDP per capita (USD)"]].values
y = df[["Life satisfaction"]].values

# visualizing the data
df.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction')
plt.axis([10_000, 100_000, 0, 12])
plt.show()

model = LinearRegression()
model.fit(x, y)

prediction = model.predict([[37_655.2]])
print(prediction)