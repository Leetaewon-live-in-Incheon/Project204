/*
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print "x:\n" , x


# In[5]:


import matplotlib.pyplot as plt
x = np.linspace(-10, 10, 100)

y = np.sin(x)

plt.plot(x,y, marker='x')
plt.show()


# In[7]:


import numpy as np
print("Hello, world")

x=3
print("����: %01d, %02d, %03d, %04d, %05d" % (x,x,x,x,x))

x = 256.123
print("�Ǽ�: %.0f, %.1f, %.2f" % (x,x,x))

x = "Hello World"
print("���ڿ�: [%s]" % [x])
*/

/*
#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd


# In[9]:


# �ӽŷ��װ� ��� �о߿��� ���� ������ ����� �� �ײ�(iris) �����ͼ�
from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[10]:


print"iris_dataset�� Ű:\n", iris_dataset.keys()


# In[11]:


print iris_dataset['DESCR'][:193] + "\n..."


# In[12]:


print "Ÿ���� �̸�:", iris_dataset['target_names']


# In[13]:


print"Ư���� �̸�:\n", iris_dataset['feature_names']


# In[14]:


print"data�� Ÿ��:", type(iris_dataset['data'])


# In[15]:


print"data�� ũ��:", iris_dataset['data'].shape


# In[16]:


print"data�� ó�� �ټ� ��:\n", iris_dataset['data'][:5]


# In[17]:


print"target�� Ÿ��:", type(iris_dataset['target'])


# In[18]:


print"target�� ũ��:", iris_dataset['target'].shape


# In[21]:


print"Ÿ��:\n", iris_dataset['target']


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(iris_dataset['data'],
                                                     iris_dataset['target'],
                                                     random_state=0)


# In[32]:


print"X_train ũ��:", X_train.shape
print"y_train ũ��:", y_train.shape


# In[33]:


print"X_test ũ��:", X_test.shape
print"y_test ũ��:", y_test.shape


# In[40]:


# X_train �����͸� ����ؼ� �������������� ����.
# ���� �̸��� iris_dataset. feature_names�� �ִ� ���ڿ��� ���
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# �������������� ����� y_train�� ���� ������ ���е� ������ ����� ����.
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=8)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[42]:


knn.fit(X_train, y_train)


# In[49]:


X_new = np.array([[5, 2.9, 1, 0.2]])
print "X_new.shape", X_new.shape


# In[51]:


prediction = knn.predict(X_new)
print "����:", prediction
print "������ Ÿ���� �̸�:", iris_dataset['target_names'][prediction]


# In[52]:


y_pred = knn.predict(X_test)
print "�׽�Ʈ ��Ʈ�� ���� ������:\n", y_pred


# In[53]:


print "�׽�Ʈ ��Ʈ�� ��Ȯ��: {:.2f}".format(np.mean(y_pred == y_test))
*/

/*
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# (�� ũ��(��)),(����(���))
# �б���ó(ȫ����,ȫ����,���⵿,�ſ���,��â��,�����)

import pandas as pd
data = pd.read_csv('house_price.txt', names=['size','price'])
print data

X = data['size']
y = data['price']
m = len(data)


# In[21]:


# numpy array ���·� ��ȯ, ���� ��ȯ(m) -> (m,1)
X = (np.array(X)).reshape(m,1)
y = (np.array(y)).reshape(m,1)
print X.shape, y.shape


# In[26]:


import matplotlib.pyplot as plt
plt.plot(X, y, 'b.')
plt.xlabel("Size of hous in Pyeong") # �� ũ��(��)
plt.ylabel("Price in uck-won") # �ŸŰ�(���)
plt.show()


# In[27]:


X_b = np.c_[np.ones((m, 1)), X] # ��� ���ÿ� x0=1�� �߰�
#c_ concatenation. �迭�� ������ ���̱�

learning_rate = 0.0001 #�н���(learning rate)
n_iter = 200

theta = np.zeros((2,1))
gradients = np.zeros((2,1))

for i in range(n_iter):
    gradients = 2.0/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
print "theta:"
print theta


# In[33]:


X_new = np.array([[10], [90]]) #10~90��
X_new_b = np.c_[np.ones((2,1)), X_new] # ��� ���ÿ� x0=1�� �߰�
y_predict = X_new_b.dot(theta)
print "10��� 90�� ���� ���� ����"
print y_predict


# In[29]:


plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.xlabel("Size of a house in Pyeong") # �� ũ��(10��)
plt.ylabel("Price in uck-won") # �ŸŰ�(���)
plt.show()


# In[30]:


X_mine = np.array([[25]]) # �츮 ���� 25���̶��, ���ϱ�?
X_mine_b = np.c_[np.ones((1,1)), X_mine] # ��� ���ÿ� x0=1�� �߰�
y_predict = X_mine_b.dot(theta)
print y_predict, "���"


# In[34]:


from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=100, penalty=None, eta0=0.0001) # eta0(learning rate)
sgd_reg.fit(X, y.ravel()) #sgd_reg.fit(X, y.ravel()) # 1���� �迭�� ����� (m,1) ->(m)


# In[32]:


print "theta"
print sgd_reg.intercept_, sgd_reg.coef_

y_pred = sgd_reg.predict(X)
plt.plot(X, y, 'b.')
plt.plot(X, y_pred, 'r')
plt.show()


# In[35]:


# �츮���� 25���̶�� ���ϱ�?
y_pred = sgd_reg.predict([[25]])
print y_pred, "���"


# In[36]:


from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# In[37]:


def ComputeJ(t0, t1, X, y):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X] # ��� ���ÿ� x0=1�� �߰�
    theta = np.array([t0, t1])

    cost_vector = X_b.dot(theta) - y # (52,1)

    cos_vector = cost_vector.reshape(-1) # �� ���� ����
    cost2 = np.square(cost_vector) # ���ҵ� ���ϱ�
    cost_sum = np.sum(cost2) / (2*m)


    return cost_sum


# In[40]:


# 1. theta0, theta1, J �� ���ϱ�
d = 100 # # of split

s=1.0
theta0 = np.linspace(-15*s, 15*s, d) # -0.1 ~ 0.1
theta1 = np.linspace(-1.0*s, 1.0*s, d) # 0.1 ~ 0.2

J = np.ones((d,d)) # (100,100)
for i,t0 in enumerate(theta0):
    for j,t1 in enumerate(theta1):
        cost = ComputeJ(t0,t1, X, y)
        J[i,j] = cost


# In[41]:


# 2. �׸���
pX, pY = np.meshgrid(theta0, theta1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(pX, pY, J, 300, cmap='viridis')
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J');
plt.show()
*/

/*

*/

#include <stdio.h>
int main(void) {
    printf("Hello World!\n");

}