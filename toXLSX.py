from pyexcel.cookbook import merge_all_to_a_book
import pyexcel.ext.xlsx # needed to support xlsx format, pip install pyexcel-xlsx
import glob


merge_all_to_a_book(glob.glob("result/*.csv"), "output.xlsx")
