from requests import get
from pandas import read_json, options
from json import loads, dumps
from numpy import array, concatenate
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import joblib

data = get('https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=TS,CLRSKY_DAYS,CLOUD_AMT,CLOUD_OD&community=RE&longitude=-56.0840&latitude=-15.5702&format=JSON&start=2010&end=2019')
target = get('https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=DIRECT_ILLUMINANCE&community=RE&longitude=-56.0840&latitude=-15.5702&format=JSON&start=2010&end=2019')

data = data.json()
target = target.json()

coords = data['geometry']['coordinates'][:-1]
coords = coords[::-1]

data = data['properties']['parameter']
target = target['properties']['parameter']

data = read_json(dumps(data))
target = read_json(dumps(target))

# Tranformando para JSON com orientação "split"
dt = data.to_json(orient='split')
tg = target.to_json(orient='split')

dt = loads(dt)
tg = loads(tg)

ta = []
for t in tg['data']:
	ta.append(t[0])

tg['data'] = ta

dict_tg = dict()
dict_tg['target'] = array(tg['data'])

dict_dt = dict()
dict_dt['data'] = array(dt['data'])

X, Y = resample(dict_dt['data'], dict_tg['target'], random_state=0)

target = array([])
for y in Y:
	y = y/1000
	if(y >= 0 and y <= 30):
		target = concatenate((target, [1]), axis=0)
	elif(y > 30 and y <= 40):
		target = concatenate((target, [2]), axis=0)
	elif(y > 40 and y <= 60):
		target = concatenate((target, [3]), axis=0)
	elif(y > 60 and y <= 80):
		target = concatenate((target, [4]), axis=0)
	elif(y > 80):
		target = concatenate((target, [5]), axis=0)

clf = DecisionTreeClassifier()
clf.fit(X, target)
clf.score(X, target)
joblib.dump(clf, 'decision_tree.pk1')