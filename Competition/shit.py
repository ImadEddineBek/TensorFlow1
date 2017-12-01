# # import numpy
# # import pandas
# # from matplotlib import pyplot as plt
# #
# # data = pandas.read_csv('fav22 - Copy (2).csv')
# # names = [x for x in data.columns if x not in ['connection_id']]
# # correlations = data.corr()
# # # plot correlation matrix
# # correlations.to_csv('shit.csv')
# # data = pandas.read_csv('shit.csv')
# # for i, row in enumerate(data.values):
# #     for j, val in enumerate(row[1:]):
# #         if val > 0.8:
# #             print(val,row[0], i, j)
# s = 'cont_1,cont_2,cont_3,cont_6,cont_7,cont_8,cont_9,cont_10,cont_12,cont_13,cont_14,cat_1,cat_2,cat_3,cat_4,cat_5,cat_6,cat_7,cat_8,cat_9,cat_11,cat_12,cat_14,cat_15,cat_16,cat_17,cat_18,cat_19,cat_20,cat_21,cat_22,cat_23'
# s = s.split(',')
# l = ''
# for j in s:
#     l += "'" + j + "',"
# print(l)
import numpy
import pandas

data = pandas.read_csv('test_data.csv')
data = data['connection_id']
tab = []
for i in data:
    i = str(i).replace('cxcon_','')
    print(i)
    tab.append(i)
pandas.DataFrame(numpy.array(tab)).to_csv('bbb.csv')