import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy
import matplotlib

#om concvert excel from https://stackoverflow.com/questions/37403460/how-would-i-take-an-excel-file-and-convert-its-columns-into-lists-in-python



#Scaling, inte om det �r sparse kanske, http://scikit-learn.org/stable/modules/preprocessing.html

# Kolla också standard nr2 sclared options på https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/


from xlrd import open_workbook

book = open_workbook("test.xlsx")
sheet = book.sheet_by_index(0) #If your data is on sheet 1

column1 = []
column2 = []

column3 = []
column4 = []
column5 = []



#...

yourlastrow=14459

for row in range(1, yourlastrow): #start from 1, to leave out row 0
    column2.append(sheet.cell(row, 2))
    column3.append(sheet.cell(row, 3)) #extract from first col
    column4.append(sheet.cell(row, 4))
    #column5.append(sheet.cell(row, 5))

    #...
    

#print(len(column3))

#print(len(column2))

#print(len(column4))

#print(column2[2])

#print(column4[2])
#print(column3[2])

#print(column2[14459])
#print(column4[14459])
#print(column3[14459])


colList2=[]

colList3=[]

colList4=[]



#Transform Xlrd cell object to vlues in lists

for i in range(len(column2)):
    colList2.append(column2[i].value)
    



for i in range(len(column3)):
    colList3.append(column3[i].value)


for i in range(len(column4)):
    colList4.append(column4[i].value)



XViktRet= zip(colList2,colList3,colList4) # med tre som orginal


#XViktRet= zip(colList2,colList3)
#XViktRet=list(XViktRet)

print(int((XViktRet[5][1])))

scaler = MinMaxScaler(feature_range=(0, 1))


XTrain=[]



Xscaled2 = preprocessing.scale(XViktRet)




"""------------------ NYTT Juni 2019 Test""------------------------------"""

#Dela upp listian i predict och inte train

print(len(Xscaled2))

Xvalue=Xscaled2[0:10000]
Xvalue2=Xscaled2[10000:14458]

print(len(Xvalue2))
print("Hej")








"""------------------ NYTT Juni 2019 Test------------------------------"""

#level1 = [list(XViktRet) for row in level1]


#Xscaled2 = [list(row) for row in XViktRet]

#Xscaled2=numpy.array(Xscaled2)
print(3)


# Mer generellt om LOF och novelty http://scikit-learn.org/stable/modules/outlier_detection.html
    
    
#Scaling preprocessing

# Specifict K�r LOF fr�n http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor

#print type(Xscaled)
#print type(Xscaled2)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

import numpy 
import scipy


#clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1 )
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(Xvalue)
y_pred_test = clf.predict(Xvalue2)

n_error_test= y_pred_test[y_pred_test == -1].size

"""Nytt"""

xx, yy ,zz = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))






y_pred_test = clf.predict(Xvalue2)


Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(),zz.ravel()])
Z = Z.reshape(xx.shape)
X_scores = clf.negative_outlier_factor_


""" end nytt 2019 """



plt.title("Novelty Detection with LOF")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

s = 40
b1 = plt.scatter(Xvalue[:, 0], Xvalue[:, 1], c='white', s=s, edgecolors='k')
b2 = plt.scatter(Xvalue2[:, 0], Xvalue2[:, 1], c='blueviolet', s=s,
                 edgecolors='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([a.collections[0], b1, b2],
           ["learned frontier", "training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left",
           prop=matplotlib.font_manager.FontProperties(size=11))
plt.xlabel(
    "errors novel regular: %d/40"
    % (n_error_test ))
plt.show()



"""
X_scores = clf.negative_outlier_factor_

import xlwt
from tempfile import TemporaryFile
book = xlwt.Workbook()
sheet1 = book.add_sheet('sheet1')


for i,e in enumerate(X_scores):
    sheet1.write(i,1,e)

name = "random_un normalized1.xls"
book.save(name)
book.save(TemporaryFile())


plt.title("Local Outlier Factor (LOF)")
plt.scatter(Xscaled2[:, 0], Xscaled2[:, 1], color='k', s=3., label='Data points')


for i, j in enumerate(Xscaled2):
    
    plt.annotate(str(i+1),xy=(Xscaled2[i][0],Xscaled2[i][1]))   


radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(Xscaled2[:, 0], Xscaled2[:, 1], s=1000 * radius, edgecolors='r',
            facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()

"""