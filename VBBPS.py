from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier,VotingClassifier,RandomForestClassifier
mc1 = AdaBoostClassifier(n_estimators=750,estimator=RandomForestClassifier())
paramsC = [{"learning_rate":[0.1,0.5,1,2],"n_estimators":[250,500,750],"random_state":[None,42]}]
mc1.get_params()
modelsimplexClassification = RandomizedSearchCV(mc1,paramsC,cv=3,scoring="precision",return_train_score=True)
m1 = GradientBoostingRegressor(n_estimators=750)
params = [{"learning_rate":[0.1,0.5,1,2],"max_depth":[3,4,5],"random_state":[None,42]}]
modelSimplexRegression=RandomizedSearchCV(m1,params,cv=3,scoring="neg_root_mean_squared_error",return_train_score=True)
# NUERAL NETWROK

m2 = VotingRegressor(estimators=[("m1",modelSimplexRegression),("m2",modelSimplexRegression)])
m_2=VotingRegressor(estimators=[("mo1",m2),("mo2",m2)])
m__2=VotingRegressor(estimators=[("moo1",m_2),("moo2",m_2)])
m___2=VotingRegressor(estimators=[("mooo1",m__2),("mooo2",m__2)])
modelNeuralNetwork = Pipeline([
    ("l1",m2),
    ("l2",m_2),
    ("l3",m__2),
    ("l4",m___2)
])
modelComplexRegression =modelNeuralNetwork
m3 = VotingClassifier(estimators=[("m1",modelsimplexClassification),("m2",modelsimplexClassification)])
m_3=VotingClassifier(estimators=[("mo1",m3),("mo2",m3)])
m__3=VotingClassifier(estimators=[("moo1",m_3),("moo2",m_3)])
m___3=VotingClassifier(estimators=[("mooo1",m__3),("mooo2",m__3)])
modelNeuralNetworkClassification = Pipeline([
    ("l1",m3),
    ("l2",m_3),
    ("l3",m__3),
    ("l4",m___3)
])
modelComplexClassification = modelNeuralNetworkClassification