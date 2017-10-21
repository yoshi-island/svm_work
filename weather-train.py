import pandas as pd
from sklearn import svm, metrics, cross_validation

csv = pd.read_csv('data.csv')

csv_data = csv[["temp","rain_mm","sun_h","snow_cm","wind_ms","cloud_percent"]]
csv_label = csv["daytime"]

train_data, test_data, train_label, test_label = cross_validation.train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print('accuracy: ', ac_score)

print(pre)
