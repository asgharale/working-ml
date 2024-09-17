import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# EXAMPLE DATA
points = {
	"blue": [
		(2,4), (1,3), (2,3), (3,2), (2,1)
	],
	"green": [
		(5,6), (4,5), (4,6), (6,6), (5,4)
	]
}

# TEST DATA
test = [
	(3,3), (4,3)
]



# distance function
def euclidean_distance(p1, p2):
	return np.sqrt(np.sum(np.array(p1) - np.array(p2)) ** 2)


class KNearestNeighbors:


	def __init__(self, k=3):
		self.k = k
		self.pre = None
		self.near_neighbors = []

	def fit(self, points):
		self.points = points

	def predict(self, new_point):
		neighbors = []

		for cat in points:
			for point in points[cat]:
				distance = euclidean_distance(point, new_point)
				neighbors.append([distance, cat])

		self.near_neighbors = [point[1] for point in sorted(neighbors)][:self.k]

		result = Counter(self.near_neighbors).most_common(1)[0][0]
		return result



clf = KNearestNeighbors()
clf.fit(points)

predict = clf.predict(test[0])
print(predict)