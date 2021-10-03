from flask import Flask, jsonify, request

# [1] importo o deserializador
import joblib 

# [2] Carrego a classe de predição do diretório local
clf = joblib.load('decision_tree.pk1')


print(clf)
app = Flask(__name__)

@app.route('/')
def flowers_predictor():

	# parametros TS,CLRSKY_DAYS,CLOUD_AMT,CLOUD_OD

	TS = float(request.args.get('TS'))
	CLRSKY_DAYS = float(request.args.get('CLRSKY_DAYS'))
	CLOUD_AMT = float(request.args.get('CLOUD_AMT'))
	CLOUD_OD = float(request.args.get('CLOUD_OD'))

	event = [TS, CLRSKY_DAYS, CLOUD_AMT, CLOUD_OD]
	target_names = ['0ate30', '30ate40', '40ate60','60ate80', '80maior']

	# print(event, target_names)

	result = {}

	# [4] Realiza predição com base no evento
	prediction = clf.predict([event])[0]

	# print(prediction)
# 
	# [5] Realizar probabilidades individuais das 5 classes
	probas = dict(zip(target_names, clf.predict_proba([event])[0]))

	print(probas)

	# # [6] Recupera o nome real da classe
	result['prediction'] = prediction
	result['probas'] = probas

	return jsonify(result), 200

app.run()