import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
%matplotlib inline

data = pd.read_csv('mnist_test.csv')
p = data.iloc[4,1:].values
p = p.reshape(28,28).astype('uint8')
plt.imshow(p)

df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

#creating test and train size of the model
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size = 0.2,random_state = 4)
y_train.head()
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
op = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == op[i]:
        count+=1
print(count)
print("Accuracy:",count/len(pred)
