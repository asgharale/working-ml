import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv", sep=";")
df = df[["studytime", "G3"]]

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].G3
        total_error += (y - (m * x + b)) ** 2
    total_error /= float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].G3
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    
    return m, b

m = 0
b = 0
L = 0.1
ep = 200

for i in range(ep):
    m, b = gradient_descent(m, b, df, L)

print(m, b)

plt.scatter(df.studytime, df.G3)
plt.plot(list(range(0,8)), [m*x+b for x in range(8)], color="red")
plt.show()