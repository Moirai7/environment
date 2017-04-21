#-- coding:utf-8 --
import pandas as pd
import sys

def readCSV(filename):
        return pd.read_csv(filename,sep=' ')

if __name__ == '__main__':
	try:
		filename1 = sys.argv[1]
		filename2 = sys.argv[2]
	except:
                #raise
                print "Unexpected error:", sys.exc_info()[0]
	data1 = readCSV(filename1)
	data2 = readCSV(filename2)
	data = pd.merge(data1,data2,on='ID')
	data.to_csv('newdata.csv',index = False)
