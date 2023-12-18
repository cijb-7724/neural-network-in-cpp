from sklearn import datasets
import csv
import pprint
iris = datasets.load_iris()

print(iris.target_names)
print(iris.feature_names)
X = iris.data
y = iris.target
print(X)
print(y)

l=[[1, 2, 3],[4, 5, 6], [3,4,5,1]]
with open('data/iris_datasets_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(X)
with open('data/iris_datasets_target.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(y)