#--coding:utf-8
 
#regex = ['Light']
#regex = ['Light','SO2','NO2']
#regex = ['Light','SO2','NO2','LSTV','NPP']
regex = ['NPP','Light','SO2','NO2','LSTV','X','Y']
regex2 = ['cluster','NPP','Light','SO2','NO2','LSTV','X','Y']


from sklearn import linear_model
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
#'''
#线性回归+贝叶斯+随机森林+多项式
#clf = linear_model.LinearRegression()
#clf = linear_model.BayesianRidge()
clf = ensemble.RandomForestRegressor(n_estimators=200,max_depth=None,max_features=4,oob_score=False,random_state=531)
#clf = ensemble.GradientBoostingRegressor(n_estimators=2000,max_depth=7,learning_rate=0.01,subsample=0.5,loss='ls')
#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', linear_model.LinearRegression(fit_intercept=False))])
#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', ensemble.RandomForestRegressor(n_estimators=200,max_depth=None,max_features=4,oob_score=False,random_state=531))])
#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', linear_model.BayesianRidge())])
#clf = ensemble.AdaBoostRegressor(linear_model.BayesianRidge(),n_estimators=300, random_state=np.random.RandomState(1))
#clf = ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth=7),n_estimators=300, random_state=np.random.RandomState(1))
#clf = DecisionTreeRegressor(max_depth=7)
#'''

CLUNUM = 10

SPLIT = True
