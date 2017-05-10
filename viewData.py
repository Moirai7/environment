# -*- coding: utf-8 -*- 
import sys
import pandas as pd
import matplotlib 
matplotlib.use('PS')
import  matplotlib.pyplot as plt
import numpy as np
from conf import *
from scipy.stats import gaussian_kde
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error,r2_score
#import tensorflow as tf

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
	#x = []
	#for t,h in zip(yTest,yhat):
	#	#print t,h
	#	if t == 0:
	#		x.append(h/(t+1))
	#	else:
	#		x.append(h/t)
	#plt.hist(x,bins=20,normed=True)
	#x = xrange(0,len(yTest))
	x=xrange(0,25000)
	plt.plot(x,yTest[:25000])
	plt.plot(x,yhat[:25000])
	plt.show()
	pass
''''
# Create model
def multilayer_perceptron(x,n_input):
	# Network Parameters
	n_hidden_1 = 32 # 1st layer number of features
	n_hidden_2 = 200 # 2nd layer number of features
	n_hidden_3 = 200
	n_hidden_4 = 256
	n_classes = 1
	
	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.01)),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.01)),
		'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.01)),
		'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.01)),
		'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.01))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.01)),
		'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.01)),
		'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.01)),
		'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.01)),
		'out': tf.Variable(tf.random_normal([n_classes], 0, 0.01))
	}

	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)

	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)

	# Hidden layer with RELU activation
	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.relu(layer_3)

	# Hidden layer with RELU activation
	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
	layer_4 = tf.nn.relu(layer_4)

	# Output layer with linear activation
	#out_layer = tf.transpose(tf.matmul(layer_4, weights['out']) + biases['out'])
	out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
	return out_layer
def usingMLP(df):
	from tensorflow.contrib import learn
	X = df[regex2]
	y = df['TE']
	xTrain2,xTest2,yTrain,yTest = train_test_split(X,y,test_size=0.30,random_state=531)
	if SPLIT:
		xTrain = preprocessing.scale(xTrain2[regex])
		xTest = preprocessing.scale(xTest2[regex])
	else:
		xTrain = preprocessing.scale(X[regex])
		xTest2 = xTrain
		xTest = xTrain
		yTrain = y
		yTest = y
	yTrain = yTrain.as_matrix()
	yTest = yTest.as_matrix()
	yTrain = yTrain.reshape((len(yTrain),1))
	yTest = yTest.reshape((len(yTest),1))
	# Parameters
	learning_rate = 0.1
	training_epochs = 10
	batch_size = 100
	display_step = 1
	dropout_rate = 0.9
	total_len = xTrain.shape[0]
	n_input = xTrain.shape[1]

	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, 1])
	
	# Construct model
	pred = multilayer_perceptron(x,n_input)
	# Define loss and optimizer
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = tf.reduce_mean(tf.sqrt(tf.abs(pred*pred-y*y)))
	#cost = tf.reduce_mean(tf.square(pred-y))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	# Launch the graph
	with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())

	    # Training cycle
	    for epoch in range(training_epochs):
		training_batch = zip(range(0, len(xTrain), batch_size),
                             range(batch_size, len(xTrain)+1, batch_size))	
		epoch_loss = 0
	        # Loop over all batches
		for start, end in training_batch:
	            # Run optimization op (backprop) and cost op (to get loss value)
	            _, c = sess.run([optimizer,cost], feed_dict={x: xTrain[start:end],y: yTrain[start:end]})
		    epoch_loss += c
	        print('Epoch', epoch, 'completed out of', training_epochs, 'loss:', epoch_loss)

	    print ("Optimization Finished!")

	    # Test model
	    correct_prediction = tf.equal(tf.argmax(pred, 1),  tf.argmax(y,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	    print sess.run(accuracy, feed_dict={x: xTest, y: yTest})
	    yhat = sess.run(pred,feed_dict={x: xTest, y: yTest})
	    print "MSE:",str(mean_squared_error(yTest,yhat))
            print 'R-squared:',str(r2_score(yTest,yhat))
	    return (xTest2,yTest,yhat)
'''
def regression(df):
	X = df[regex2]
	y = df['TE']
	xTrain2,xTest2,yTrain,yTest = train_test_split(X,y,test_size=0.30,random_state=531)
	if SPLIT:
		xTrain = preprocessing.scale(xTrain2[regex])
		xTest = preprocessing.scale(xTest2[regex])
	else:
		xTrain = preprocessing.scale(X[regex])
		xTest2 = xTrain
		xTest = xTrain
		yTrain = y
		yTest = y
	yTrain = yTrain.as_matrix()
        yTest = yTest.as_matrix()
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
	clf.fit(xTrain,yTrain)
	yhat = clf.predict(X = xTest)
	#print clf.intercept_,
	#print clf.coef_
	#print clf.named_steps['linear'].coef_
	print "MSE:",str(mean_squared_error(yTest,yhat))
	print 'R-squared:',str(r2_score(yTest,yhat))
	#'''
	if printTree:
		from sklearn import tree
		i_tree = 0
		for tree_in_forest in clf.estimators_:
    			with open('result/tree_' + str(i_tree) + '.dot', 'w') as my_file:
        			my_file = tree.export_graphviz(tree_in_forest, out_file = my_file)
    			i_tree = i_tree + 1
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
	X = df[regex]
	yhat = KMeans(n_clusters=CLUNUM, init='k-means++', max_iter=300, n_init=1,verbose=False).fit_predict(X)
	
	#plt.scatter(df['X'],df['Y'],c=yhat)
	#plt.show()
	return yhat

def proc1(data,code,pclusters,ppred):	
	#regression(data)
	#clusters_test(data)
	data['cluster'] = clusters(data)
	pclusters.append(data)
	cluster = data.drop_duplicates(['cluster'])['cluster']
	yTest = []
	yhat = []
	xTest = []
	for c in cluster:
		print "###################"
		print "count(cluster): ",str(len(data[data.cluster==c]))
		if len(data[data.cluster==c])<10:
			continue
		#showInfo(data[data.cluster==c],x,y)
		#x,t,p = regression(data[data.cluster==c])
		x,t,p = usingMLP(data[data.cluster==c])
		printTree = False
		xTest.append(pd.DataFrame(x))
		yTest.append(pd.DataFrame(t))
		yhat.append(pd.DataFrame(p))
	xTest = pd.concat(xTest,ignore_index=True)
	yTest = pd.concat(yTest,ignore_index=True)
	yhat = pd.concat(yhat,ignore_index=True)
	res = xTest
	res['code'] = code
	res['yTest'] = yTest
	res['yPred'] = yhat
	ppred.append(res)
	print "###################"
	print "MSE:",str(mean_squared_error(yTest,yhat))
        print 'R-squared:',str(r2_score(yTest,yhat))

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
	#data[data.TE==0.0]['TE'] = 0.00001

	if info:
		showInfo(data,x,y)
	if graph:
		oneGraph(data,x)
	if paired:
		pairedGraph(data,x,y)
	if z:
		drawMap(x,y,z)
	try:
		pclusters = []
		ppred = []
		code = data.drop_duplicates(['CODE'])['CODE']
		for c in code:
			print '\n\ncode : '+str(c),str(len(data[data.CODE==c]))
			if len(data[data.CODE==c])<5:
				continue
			proc1(data[data.CODE==c],c,pclusters,ppred)
			
	except:
		#print "Unexpected error:", sys.exc_info()[0]
		proc1(data,0,pclusters,ppred)
	ppred = pd.concat(ppred)
	pclusters = pd.concat(pclusters)
	pclusters.to_csv('result/cluster.csv',  index = False)
	ppred.to_csv('result/pred.csv',  index = False)
	showRes(ppred['yTest'],ppred['yPred'])
