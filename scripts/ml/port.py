import joblib

from sklearn_porter import Porter

model = joblib.load('models/best_model_DecisionTree.joblib')
Porter(model, language='c', method='predict')

# print(est.port())
#
# est.export(directory="../../../src/")
