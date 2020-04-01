import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# %% LOAD THE DATA
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])
df.head()
#print(df)

#Pre-processing the data
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, ['target']].values

# performs standardization
x = StandardScaler().fit_transform(x)
pd.DataFrame(data=x, columns=features).head()
#print(pd)

#calculation of covariance matrix
mean_vec = np.mean(x, axis=0)
mean_remv_x = x - mean_vec
cov_mat = (x - mean_vec).T.dot((x - mean_vec))/(x.shape[0]-1)
#print('Covariance matrix \n %s' %cov_mat)

#Eigen decomposition on covariance matrix
cov_mat = np.cov(x.T)
eg_vals, eg_vecs = np.linalg.eig(cov_mat)
#print('EigenVectors \n%s' %eg_vecs)
#print('EigenValues \n%s' %eg_vals)

#computing the Principal components
eg_vecs_trunc = eg_vecs[:,:2]
eg_vecs_trunc_trans = np.transpose(eg_vecs_trunc)
new_dim_x = np.dot(eg_vecs_trunc_trans, np.transpose(mean_remv_x))
principalComponents = np.transpose(new_dim_x)
#print('Principal Components::\n')
#print(principalComponents)

principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
principalDf.head(5)
#print(principalDf)

df[['target']].head()
#print(df[['target']])

finalDf = pd.concat([principalDf, df[['target']]], axis=1)
finalDf.head(5)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=color
               , s=50)
ax.legend(targets)
ax.grid()
plt.show()
#print(pca.explained_variance_ratio_)
