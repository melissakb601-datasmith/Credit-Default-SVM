import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data into a 2D array
ccdata = np.loadtxt("C:\Users\Path_to\ccdata.txt", delimiter=",", skiprows=1, dtype="str")

# get domain and range values
DomainVal = ccdata[:,2:24]
DomainVal = DomainVal.astype(float)
RangeVal = ccdata[:,24:25]
RangeVal = RangeVal.flatten()

s = StandardScaler()
DomainVal = s.fit_transform(DomainVal)

x_train, x_test, y_train, y_test = train_test_split(DomainVal, RangeVal, test_size=0.25)
y_train = y_train.flatten()
y_test = y_test.flatten()

svm = SVC()
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)
print("Accuracy", accuracy_score(y_test, y_predict))
