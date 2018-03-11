import sys
import numpy as np

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

def save(filename,data,title):
        f = open(filename,'w')
        f.write('\r\n'.join(title)+'\r\n')
        res = []
        for d in data:
                d = ['{:.6f}'.format(x) for x in d]
                res.append(' '.join(d))
        data = ' \r\n'.join(res)
        f.write(data)

def confine(data,maxs=1.1):
	data[data>maxs]=maxs
	return data

filename = 'pure_s_sr.txt'
data,title = read_data(filename)
data = confine(data)
save(filename+'2',data,title)
