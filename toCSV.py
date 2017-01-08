import xlrd
import csv
import sys

def csv_from_excel(filename,sheet):
    wb = xlrd.open_workbook('raw/'+filename)
    sh = wb.sheet_by_name(sheet)
    your_csv_file = open('data/'+filename+'_'+sheet+'.csv', 'wb')
    wr = csv.writer(your_csv_file)#, quoting=csv.QUOTE_ALL)

    for rownum in xrange(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()

if __name__ == '__main__':
	if (len(sys.argv)==3) :
		csv_from_excel(sys.argv[1],sys.argv[2])
	elif (len(sys.argv)==2) :
		csv_from_excel(sys.argv[1],"Sheet1")
	else:
		print 'Wrong!'
