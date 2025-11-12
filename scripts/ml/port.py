import joblib

from sklearn_porter import Estimator

model = joblib.load('models/best_model_DecisionTree.joblib')
est = Estimator(model, language='c', template='attached')

print(est.port())

est.template = 'exported'
est.save(directory="../../../src/")
