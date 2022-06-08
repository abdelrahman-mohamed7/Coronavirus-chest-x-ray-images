import numpy as np
import pandas as pd
from PIL import Image
import os 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

path_N = "D:\\fci-cu\\third year\\first\\machine\\assignment2\\Assignment 2 - Dataset\\Assignment 2 - Dataset\\Dataset\\Negative"
path_P = "D:\\fci-cu\\third year\\first\\machine\\assignment2\\Assignment 2 - Dataset\\Assignment 2 - Dataset\\Dataset\\Positive"

# function to convert images into binary values (only 0 and 255  "black and white") then turn it into dataframe
def bin_images(path,t,p,thresh = 130):
    chestdata =[]
    target = []
    for j,i in enumerate(os.listdir(path)):
        image_gray = np.array(Image.open(path+"\\"+i).convert('L'))
        threshold = thresh
        maxvalue = 255
        image_bin = (image_gray > threshold) * maxvalue
        chestdata.append(image_bin.flatten())
        target.append(t)
        Image.fromarray(np.uint8(image_bin)).save("D:\\fci-cu\\third year\\first\\machine\\assignment2\\Assignment 2 - Dataset\\Assignment 2 - Dataset\\Dataset\\"+str(j+1)+p+'.jpg')
    df = pd.DataFrame(chestdata)
    df['target'] = target
    return df



# converting the data into dataframe
q = input("do you want to enter sepcific threshold?(y \ n)")
if(q == 'y' or q == 'Y'):
    thre = int(input("enter the threshold : "))
    N_df = bin_images(path_N,0,'N',thre)
    P_df = bin_images(path_P,1,'P',thre)
else:
    N_df = bin_images(path_N,0,'N',thre)
    P_df = bin_images(path_P,1,'P',thre)



# combine them in one dataframe
df = P_df.append(N_df)
df = df.reset_index(drop=True)



# seperating data to apply the model on it
x = df.drop(['target'] , axis = 1)
y = df['target']
Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.30)



# using svm model with no grid function(default hyper-parameters)
classfication_with_nogrid = SVC()
classfication_with_nogrid.fit(Xtrain, Ytrain)
Ypred_with_nogrid = classfication_with_nogrid.predict(Xtest)
print(f'the accuracy of the model (which has no grid (default hyper-parameters)) is : {accuracy_score(Ytest, Ypred_with_nogrid) * 100} %')



# using svm model with grid function(different hyper-parameters)
# C : controls the error -> lower c mean less error
# gamma : controls the curvature of the decision boundary --> high gamma means more curvature.(mainly affects rbf)
# trying different hyper-parameters to get good results
gridparameters={'C':[0.1,1,10,100],'gamma':[1, 0.1, 0.01, 0.001, 0.0001],'kernel':['rbf','poly','linear','sigmoid']}
classfication_with_grid=GridSearchCV(SVC(),gridparameters)
classfication_with_grid.fit(Xtrain, Ytrain)
Ypred_with_grid = classfication_with_grid.predict(Xtest)
print(f'the accuracy of the model (which has grid (different hyper-parameters)) is : {accuracy_score(Ytest, Ypred_with_grid) * 100} %')




print("the parameters were chosen by the grid are {} " .format(classfication_with_grid.best_params_) )




