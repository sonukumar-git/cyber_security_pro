from django.shortcuts import render

# Create your views here.

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as spl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#import library to measurement
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import seaborn as sb

from django.http import HttpResponse
from django.shortcuts import render


data=pd.read_csv("static/data/Train_data.csv")

data.head()
unwanted_col=['wrong_fragment','urgent','num_failed_logins','num_file_creations','num_shells','num_outbound_cmds']

df=data.drop(unwanted_col,axis='columns')

df.protocol_type.value_counts()
x={'tcp':0,'udp':1,'icmp':2}
new_col=[ x[items]  for items in df.protocol_type ]
df.protocol_type=new_col


df.service.value_counts()
lb=LabelEncoder()
lb.fit(df.service)  #fiting the column
new_col=lb.transform(df.service)  #transforming the column
df.service=new_col


df.flag.value_counts()
df.flag=lb.fit_transform(df.flag)


df.duration=np.where((df.duration>=2),0,1)


f=sb.countplot(x='service',hue='class',data=df)
fig=f.get_figure()
fig.savefig('static/service.png')


#now feature and label extraction
#last column is the label and remaining are the feature to find it
#slice it to X(feature),y(label)
#using dataframe.iloc[starting row index:ending row,starting column index:ending column]

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

#X
#y


def index(request):
    #return HttpResponse("Home")
    return  render(request,'home.html')

def logistic(request):
    Xtrain,Xtest,ytrain,ytest=spl(X,y,test_size=0.2)
    lr=LogisticRegression() #LogisticRegression object
    lr.fit(Xtrain,ytrain) #training the algorithm

    pred_lr=lr.predict(Xtest)
    #making dataframe for comparision
    df_lr=pd.DataFrame()
    df_lr['ytest']=ytest
    df_lr['prediction']=pred_lr
    cm=confusion_matrix(ytest,pred_lr)
    a=accuracy_score(ytest,pred_lr)
    parameter1={'acc':a,'confusion':cm}
   
   
    
    return render(request,'logistic.html',parameter1)

def random(request):
    Xtrain,Xtest,ytrain,ytest=spl(X,y,test_size=0.2)
    rf=RandomForestClassifier()
    rf.fit(Xtrain,ytrain)
    pred_rf=rf.predict(Xtest)
    
    cm=confusion_matrix(ytest,pred_rf)
    a=accuracy_score(ytest,pred_rf)
    parameter1={'acc':a,'confusion':cm}
     
    return render(request,'random.html',parameter1)

def test1(request):
    return render(request,'test.html')
