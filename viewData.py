import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scipy.stats as stats
#from mpl_toolkits.basemap import Basemap
import folium

def readCSV(filename,index,header):
	return pd.read_csv(filename, sep=',', engine='python', header=header, index_col=index)

def showInfo(data,x,y):
	data.info()
	print data.head()
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

def regression(df):
	from sklearn import linear_model
	from sklearn.metrics import mean_squared_error,r2_score
	#regex = ['Light']
	#regex = ['Light','SO2','NO2']
	regex = ['Light','SO2','NO2','LSTV','NPP']
	#regex = ['Light','SO2','NO2','X','Y','LSTV','NPP']
	X = df[regex]
	y = df['TE']
	'''
	lr = linear_model.LinearRegression()
	lr.fit(X,y)
	yhat = lr.predict(X = df[regex])
	#print lr.intercept_,lr.coef_
	print "MSE:",str(mean_squared_error(df['TE'],yhat))
	
	import statsmodels.formula.api as sm
	linear_model = sm.OLS(y,X)
	results = linear_model.fit()
	print results.summary()
	'''
	'''
	reg = linear_model.BayesianRidge()
	reg.fit(X, y)
	yhat = reg.predict(X = df[regex])
	print reg.intercept_,reg.coef_
	print "MSE:",str(mean_squared_error(df['TE'],yhat))
	print 'R-squared:',str(r2_score(df['TE'],yhat))
	'''
	
	pass

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
	data = data[data.TE!=0]

	if info:
		showInfo(data,x,y)
	if graph:
		oneGraph(data,x)
	if paired:
		pairedGraph(data,x,y)
	if z:
		drawMap(x,y,z)
	regression(data)
