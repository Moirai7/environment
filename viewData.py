# -*- coding: utf-8 -*- 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score

def readCSV(filename,index,header):
	return pd.read_csv(filename, sep=',', engine='python', header=header, index_col=index)

def showInfo(data,x,y):
	from math import exp
	data.info()
	#print data.head()
	info = preprocessing.scale(data)
	rows = len(data.index)
	for i in xrange(rows):
		dataRow = info[i,1:8]
		label = 1.0/(1.0+exp(-info[i,0]))
		plt.plot(dataRow,color=plt.cm.RdYlBu(label),alpha=0.1)		
	plt.xticks(xrange(-1,7),data.columns)
	print data.corr()
	plt.matshow(data.corr())
	plt.show()
	#print data.iloc[:,x]
	#print data.iloc[:,y]
	pass

def pairedGraph(data,x,y):
	h = data.iloc[:,y].values
	r = data.iloc[:,x].values
	import seaborn
	#seaborn.jointplot(x="X", y="TE", data=pd.DataFrame({"TE":h,"X":r}))
	seaborn.pairplot(data, hue="TE")
	seaborn.plt.show()
	'''
	_fig,_axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
	colors = np.random.rand(len(r))
	_axes[0][0].scatter(r, h, s=100, c=colors, alpha=0.5)
	_axes[0][0].set_title('scatter')
	_axes[0][1].boxplot([r,h],patch_artist=True,vert=True)
	_axes[0][1].set_title('boxplot')
	_axes[0][2].violinplot([r,h])
	_axes[0][2].set_title('violinplot')
	_axes[1][0].hist([r,h], 20, normed=1, histtype='bar')
	_axes[1][0].set_title('hist')
	_axes[1][1].plot(xrange(0,len(r)),h,'-o')
	_axes[1][1].plot(xrange(0,len(h)),r,'-x')
	_axes[1][1].set_title('plot')
	x = xrange(0,len(h))
	density = gaussian_kde(h)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
	_axes[1][2].plot(x,density(x),'-o')
	_axes[1][2].set_title('density')
	density = gaussian_kde(r)
	density.covariance_factor = lambda : .25
	density._compute_covariance()
	_axes[1][2].plot(x,density(x),'-x')
	plt.show()
	'''
	pass

def oneGraph(data,x):
	_fig,_axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
	
	h = data.iloc[:,x].values
	x = xrange(0,len(h))
	_axes[0][0].boxplot(h,patch_artist=True,vert=True)
	_axes[0][0].set_title('boxplot')
	_axes[0][1].violinplot(h)
	_axes[0][1].set_title('violinplot')
	stats.probplot(h, dist="norm", plot=_axes[0][2])
	_axes[1][0].plot(x,h,'-o')
	_axes[1][0].set_title('plot')
	density = gaussian_kde(h)
	density.covariance_factor = lambda : .25
	density._compute_covariance()
	_axes[1][1].plot(x,density(x))
	_axes[1][1].set_title('density')
	plt.hist(h,bins=20,normed=True)
	_axes[1][2].set_title('hist')
	plt.show()

def drawMap(x,y,z):
	from mpl_toolkits.basemap import Basemap
	import folium
	h = data.iloc[:,y].values
	r = data.iloc[:,x].values
	z = data.iloc[:,z].values
	maps = folium.Map(location=[30, 0], zoom_start=2)
	for a,b,c in zip(r,h,z):
		print c
		maps.circle_marker(location=[a, b], popup=str(c))

	maps.create_map('maps/map.html')
	maps
	pass

def showRes(yTest,yhat):
	x = []
	for t,h in zip(yTest,yhat):
		#print t,h
		if t == 0:
			x.append(h/(t+1))
		else:
			x.append(h/t)
	plt.hist(x,bins=20,normed=True)
	plt.show()
	pass

def regression(df):
	from sklearn import linear_model
	from sklearn import ensemble
	from sklearn.tree import DecisionTreeRegressor
	from sklearn.cross_validation import train_test_split
	from sklearn.preprocessing import PolynomialFeatures
	from sklearn.pipeline import Pipeline
	#regex = ['Light']
	#regex = ['Light','SO2','NO2']
	#regex = ['Light','SO2','NO2','LSTV','NPP']
	regex = ['NPP','Light','SO2','NO2','LSTV','X','Y']
	regex2 = ['cluster','NPP','Light','SO2','NO2','LSTV','X','Y']
	X = df[regex2]
	y = df['TE']
	xTrain2,xTest2,yTrain,yTest = train_test_split(X,y,test_size=0.30,random_state=531)
	xTrain = preprocessing.scale(xTrain2[regex])
	xTest = preprocessing.scale(xTest2[regex])
	'''
	#线性回归
	import statsmodels.formula.api as sm
	lmodel = sm.OLS(yTrain,xTrain)
	results = lmodel.fit()
	print results.summary()
	'''
	'''
	#找随机森林参数 50，80
	nTreeList = xrange(50,250,10)
	mse = []
	for i in nTreeList:
		depth = None
		maxFeat = 4
		clf = ensemble.RandomForestRegressor(n_estimators=i,max_depth=depth,max_features=maxFeat,oob_score=False,random_state=531)
		clf.fit(xTrain,yTrain)
		yhat = clf.predict(xTest)
		mse.append(mean_squared_error(yTest,yhat))
	from matplotlib.font_manager import FontProperties
	font = FontProperties()
	alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
	plt.plot(nTreeList,mse)
	for x,y in zip(nTreeList,mse):
		t = plt.text(x, y, str(x) , fontproperties=font, **alignment)
	plt.show()
	'''
	#'''
	#线性回归+贝叶斯+随机森林+多项式
	clf = linear_model.LinearRegression()
	#clf = linear_model.BayesianRidge()
	#clf = ensemble.RandomForestRegressor(n_estimators=200,max_depth=None,max_features=4,oob_score=False,random_state=531)
	#clf = ensemble.GradientBoostingRegressor(n_estimators=2000,max_depth=7,learning_rate=0.01,subsample=0.5,loss='ls')
	#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', linear_model.LinearRegression(fit_intercept=False))])
	#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', ensemble.RandomForestRegressor(n_estimators=200,max_depth=None,max_features=4,oob_score=False,random_state=531))])
	#clf = Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', linear_model.BayesianRidge())])
	#clf = ensemble.AdaBoostRegressor(linear_model.BayesianRidge(),n_estimators=300, random_state=np.random.RandomState(1))
	#clf = ensemble.AdaBoostRegressor(DecisionTreeRegressor(max_depth=7),n_estimators=300, random_state=np.random.RandomState(1))
	#clf = DecisionTreeRegressor(max_depth=7)
	#'''
	'''
	clf.fit(X,y)
	yhat = clf.predict(X = X)
	print "MSE:",str(mean_squared_error(y,yhat))
	print 'R-squared:',str(r2_score(y,yhat))
	'''
	#'''
	clf.fit(xTrain,yTrain)
	yhat = clf.predict(X = xTest)
	#print clf.intercept_,
	#print clf.coef_
	#print clf.named_steps['linear'].coef_
	print "MSE:",str(mean_squared_error(yTest,yhat))
	print 'R-squared:',str(r2_score(yTest,yhat))
	#'''
	'''
	#算随机森林feature importance
	feature = clf.feature_importances_
	feature = feature/feature.max()
	sorted_idx = np.argsort(feature)
	barpos = np.arange(sorted_idx.shape[0])+.5
	plt.barh(barpos,feature[sorted_idx],align='center')
	plt.yticks(barpos,X.columns[sorted_idx])
	plt.show()
	'''
	return (xTest2,yTest,yhat)

def clusters_test(data):
        from sklearn.cluster import KMeans
        scores = []
	#regex = ['Light','SO2','NO2','LSTV','NPP']
	#regex = ['Light','SO2','NO2','LSTV','NPP','X','Y']
	#regex = ['X','Y']
	regex = ['TE']
	X = data[regex]
	
        for i in xrange(3,80,1):
                km = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=1,verbose=False)
                yhat = km.fit_predict(X)
                scores.append(-km.score(X)/len(X))
	for a,b in zip(scores,xrange(3,80,1)):
		print b,a
        plt.figure(figsize=(8,4))
        plt.plot(xrange(3,80,1),scores,label="error",color="red",linewidth=1)
        plt.xlabel("n_features")
        plt.ylabel("error")
        plt.legend()
        plt.show()

def clusters(df):
        from sklearn.cluster import KMeans
        scores = []
	#regex = ['Light','SO2','NO2','LSTV','NPP','X','Y']
	#regex = ['X','Y']
	regex = ['TE']
	X = data[regex]
	yhat = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=1,verbose=False).fit_predict(X)
	
	#plt.scatter(data['X'],data['Y'],c=yhat)
	#plt.show()
	return yhat

if __name__ == '__main__':
	try:
		filename = sys.argv[1]

		st = sys.argv[2].split('=')
		if st[0]=='header':
			header = 0 if int(st[1])==1 else None
		else:
			raise NameError('header')
		
		st = sys.argv[3].split('=')
		if st[0]=='index':
			index = 0 if int(st[1])==1 else False
		else:
			raise NameError('index')

		st = sys.argv[4].split('=')
		if st[0]=='info':
			info = int(st[1])
		else:
			raise NameError('info')
		
		st = sys.argv[5].split('=')
		if st[0]=='paired':
			paired = int(st[1])
		else:
			raise NameError('paired')
		
		st = sys.argv[6].split('=')
		if st[0]=='graph':
			graph = int(st[1])
		else:
			raise NameError('graph')
		
		x = int(sys.argv[7])
		y = int(sys.argv[8]) if len(sys.argv)>8 else -1
		z = int(sys.argv[9]) if len(sys.argv)==10 else 0
	except:
		#raise
		print "Unexpected error:", sys.exc_info()[0]

	plt.style.use('ggplot')
	data = readCSV(filename, index, header)
	#data[data.TE==0]['TE'] = 1

	if info:
		showInfo(data,x,y)
	if graph:
		oneGraph(data,x)
	if paired:
		pairedGraph(data,x,y)
	if z:
		drawMap(x,y,z)
	#regression(data)
	#clusters_test(data)
	data['cluster'] = clusters(data)
	data.to_csv('result/cluster.csv')
	cluster = data.drop_duplicates(['cluster'])['cluster']
	yTest = []
	yhat = []
	xTest = []
	for c in cluster:
		print "###################"
		print "count(cluster): ",str(len(data[data.cluster==c]))
		#showInfo(data[data.cluster==c],x,y)
		x,t,p = regression(data[data.cluster==c])
		xTest.append(pd.DataFrame(x))
		yTest.append(pd.DataFrame(t))
		yhat.append(pd.DataFrame(p))
	xTest = pd.concat(xTest,ignore_index=True)
	yTest = pd.concat(yTest,ignore_index=True)
	yhat = pd.concat(yhat,ignore_index=True)
	res = xTest
	res['yTest'] = yTest
	res['yPred'] = yhat
	res.to_csv('result/pred.csv')
	print "###################"
	print "MSE:",str(mean_squared_error(yTest,yhat))
        print 'R-squared:',str(r2_score(yTest,yhat))
	#showRes(yTest.values,yhat.values)
	#showRes(yTest,yhat)
	#quit(0)
