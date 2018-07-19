#encoding: utf-8
import numpy as np
import numpy.ma as ma
def read_data(filename):
	title = []
	data = []
	i=0
	with open(filename) as f:
		for line in f:
			try:
				l = line.strip(' \r\n').split(' ')
				data.append(list(map(float,l)))
			except:
				title.append(line.strip(' \r\n'))
				pass
	return np.asarray(data),title

def mean(data,m,n,no_data,ratio,types):
	xl = data.shape[0]
	yl = data.shape[1]
	
	x_offset = m/2
	y_offset = n/2
	tmp = np.copy(data)
	tmp = np.lib.pad(tmp,((x_offset,m-1-x_offset),(y_offset,n-1-y_offset)),'constant',constant_values=no_data)
	data = tmp
	res=[]
	window = np.ones((m,n))
	for i in xrange(xl):
		tmp = []
		for j in xrange(yl):
			window[0:m,0:n] = data[i:m+i,j:n+j]
			points = np.count_nonzero(window != no_data)
			if points >= (m*n)/ratio:
				if types is 'mean':
					tmp.append(np.mean(window[window!=no_data]))
				else:
					tmp.append(np.std(window[window!=no_data]))
			else:
				tmp.append(no_data)
		res.append(tmp)
	return np.asarray(res)

def my_func(x,a,b,no_data):
	odds = np.exp(a+b*x)
	return odds/(1+odds)

def my_func2(x,a,b,no_data):
	return np.power(x/a,b)

def change(data,a,b,no_data,types):
	data = ma.array(data,mask = (data==no_data))
	if types is 'mean':
		data = np.apply_along_axis(my_func,1,data,a,b,no_data)
	else:
		data = np.apply_along_axis(my_func2,1,data,a,b,no_data)
	return data.filled(fill_value=no_data)
	
def save(filename,data,title):
	f = open(filename,'w')
	f.write('\r\n'.join(title)+'\r\n')
	res = []
	for d in data:
		d = ['{:.6f}'.format(x) for x in d]
		res.append(' '.join(d))
	data = ' \r\n'.join(res)
	f.write(data)

def run():
	no_data = -9999#no_data
	ratio = 30#TODO 一个window里非no_data占的比率

	'''
	types = 'mean'#TODO 可修改成std或mean
	slopes = [45.37053597,28.12177021]#两个斜率
	intercepts = [-7.95147999,-29.84724765]#两个截距
	#filenames = ['pure_r_ndvi','pure_r_sr']#TODO 两个文件名字
	filenames = ['pure_s_ndvi','pure_s_sr']#TODO 两个文件名字
	'''

	types = 'mean'#TODO 可修改成std或mean
	slopes = [0.775,0.84]
	intercepts = [0.003,0.006]
	filenames = ['pure_r_sr_less_than_1.1','pure_s_sr_less_than_1.1']#TODO 两个文件名字
	#filenames = ['pure_r_ndvi','pure_r_sr']#TODO 两个文件名字
	#filenames = ['pure_r_ndvi','pure_r_sr']#TODO 两个文件名字

 	#windows = [33,67,133,266]#TODO 这里的windows可按照需求修改
	windows = [33]

	for filename,intercept,slope in zip(filenames,intercepts,slopes):

		print slope,intercept
		data,title = read_data(filename+'.txt')
		print 'process '+filename
		for window in windows:
			means = mean(data,window,window,no_data,ratio,types)
			strs = 'result'+str(ratio)+'/'+filename+'_w'+str(window)+'_'+types
			save(strs+'.txt',means,title)
			print strs+'.txt' +' saved!'
			#means = change(means,intercept,slope,no_data,types)
			#strs = 'result'+str(ratio)+'/'+filename+'_w'+str(window)+'_pred'
			#save(strs+'.txt',means,title)
			#print strs+'.txt'+' saved!'


if __name__ == '__main__':
	run()
