from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import pandas as pd
corpus = []
with open("D://datasets//iris//iris.data",'r') as f:
    for val in f:
        corpus.append(val.split('\n'))

refined_corpus=[]
for i in range(len(corpus)-1):
    data,_ = corpus[i];
    #print(data.split(','))
    refined_corpus.append(data.split(','))
#refined_corpus = np.array(refined_corpus)
#print(refined_corpus)
X=[];y=[]
for a,b,c,d,e in refined_corpus:
    X.append([float(a),float(b),float(c),float(d)])
    if e=='Iris-setosa':
        y.append(0);
    elif e=='Iris-versicolor':
        y.append(1);
    else:
        y.append(2);
y=np.array(y)
X=np.array(X)

model = LinearDiscriminantAnalysis(n_components=2)
X_train = model.fit_transform(X, y)
print(X_train)

data_dict = {'col1':X_train[:,0],'col2':X_train[:,1],'label':y}
df = pd.DataFrame(data_dict)
df.to_excel('D://datasets//iris//reduced_iris.xlsx')
