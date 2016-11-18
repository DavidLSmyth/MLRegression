# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:57:59 2016

@author: David Smyth
"""

#Assignment 3
#Logistic Regression Algorithm (possibly with ridge/lasso)
#Steps to achieve:
#1).Design a logReg class:
#2).Takes in a data set, a formula (predictors & target) and fits theta vector to minimise cost function
#Formula will have to allow for non-linear terms
#Gradient Descent requires learning rate, tolerance etc.
#Make this algorithm better if possible, for now just let the user pass it in
#3).Option to return theta values, std. errors would also be nice


#Details of fitting the theta vector: Gradient descent will need to be used, ideally vectorised 
#If I have a n multi-class prediction problem, then I need n models to be fitted for each of the n classes
#Details of prediction: If only two classes, straightforward. If multiclass, then return highest probability
#of results of each of the n models above
#A challenge will be to provided the right level of abstratction
#One vs. all implementation difficult
#Don't forget to add a column for the intercept

#1). Logistic regression that will ouput probabilities.
#Try plotting non-convex J(theta) cost function for better understanding

#TODO:
#1).tidy up current class
#2).Vectorise current implementation
#3).Implement multiclass 1 vs. all predictions
#4).Figure how to recover the unscaled weights
#5).Rename everything to train/test to be clear what's what


class LogisticRegression():
    def __init__(self, predictors, target, dataframe):
        import pandas as pd
        import numpy as np
        '''Pass in a list of predictors as strings, a target as a string, and a pandas dataframe'''
        #do all error checking here
        #check that the predictors & target exist in the dataframe
        if(type(dataframe)!=pd.core.frame.DataFrame):
            print('You must provide a pandas dataframe, the dataframe you have provided is of type '+type(dataframe))
            return
        #tell user which predictor(s) specifically violate this
        elif False in map(lambda x: x in dataframe, predictors):
            print('One of the predictors is not a valid column in the dataframe')
            return
        elif target not in dataframe.columns:
            print(target+' is not a valid column in the dataframe')
            return
        #if target is in the dataframe, make sure that it isn't continuous, not sure exactly how to do this
        else:
            pass
        #initally set the whole of the 
        self.TrainTestRatio=1
        self.targetName=target
        #this might not be necessary
        self.predictorNames=predictors
        #append on a column of ones
        #dataframe['intercept']=1
        self.predictors=np.matrix(dataframe[predictors],dtype=float)
        self.nsamples=np.shape(self.predictors)[0]
        self.npredictors=np.shape(self.predictors)[1]
        #np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))
        #print('predictors')
        #print(self.predictors)
        self.scaleFactor=self.predictors.max()
        self.target=np.array(dataframe[target])
        self.targetLevels=dataframe[target].unique()
        print('unique levels:')
        print(self.targetLevels)
        self.uniqueEncodedLevels=[i for i in range(len(dataframe[target].unique()))]
        print('encodedLevels')
        print(self.uniqueEncodedLevels)
        self.encodedTarget=np.array(dataframe[target])
        #self.encodedTarget=list(map(lambda x: self.uniqueEncodedLevels[list(self.targetLevels).index(x)] ,self.target))
        print('encodedTarget')
        #This treats the problem as binary for the time being
        #self.encodedTarget=list(map(lambda x: 1 if x==2 else x,self.encodedTarget))
        print(self.encodedTarget)
        self.dataframe=dataframe
        #add 1 because of the intercept value
        self.parameterVector=np.array([0 for i in range(self.predictors.shape[1]+1)])
        #self.parameterVector=[0 for i in range(len(predictors))]
        self.tolerance=0.00002
        self.learningRate=1.2
        self.maxiter=2
      
    #have multiple implementations of data preprocessing code, do it all here  
    def __preprocess(self, data):
        pass

        #Validations complete

    #fit should have 0 or 1 as the target ALWAYS and should probably take a training set to fit
    def fit(self):
        print('\n\n fit')
        print('Fitting Logistic Regression Model to ',self.TrainTestRatio,' of the dataset')
        print('length of predictors: ',len(self.predictors))
        print(self.predictors,self.encodedTarget)
        import time
        st=time.clock()
        self.__gradientDescent(self.predictors)
        print('Fit in ',time.clock()-st, ' seconds')
        print('gradient descent results: ')
        print(self.parameterVector)
    
    def trainTestSplit(self, ratio):
        '''Ratio is the desired ratio for the train set size to test set size, no return value'''
        print('\n\n trainTestSplit')
        self.TrainTestRatio=ratio
        import numpy as np
        randoms=np.random.uniform(0,1,len(self.dataframe))
        print('Random Split: ',randoms)
        self.trainSet=self.dataframe.iloc[randoms<ratio].copy(deep=True)
        self.testSet=self.dataframe.iloc[randoms>=ratio].copy(deep=True)
        #print(self.dataframe.iloc[randoms>=ratio])
        self.predictors=np.matrix(self.trainSet[self.predictorNames],dtype=float)
        self.nsamples=np.shape(self.predictors)[0]
        self.npredictors=np.shape(self.predictors)[1]
        self.encodedTarget=np.array(self.trainSet[self.targetName])
        #self.encodedTarget=list(map(lambda x: self.uniqueEncodedLevels[list(self.targetLevels).index(x)] ,self.target))
        print('encodedTarget')
        #This treats the problem as binary for the time being
        #self.encodedTarget=list(map(lambda x: 1 if x==2 else x,self.encodedTarget))
        print(self.encodedTarget)
        #add 1 because of the intercept value
        self.parameterVector=np.array([0 for i in range(self.predictors.shape[1]+1)])
        print('predictors')
        print(self.predictors)
        self.scaleFactor=self.predictors.max()
        print('length of training set: ',len(self.trainSet))
        print('length of training set: ',len(self.testSet))

    def __sigmoid(self,z):
        from math import exp
        #print('z',z)
        return 1/(1+exp(-z))
        
    def __addIntercept(self,featureMatrix):
        n_samples=np.shape(featureMatrix)[0]
        print(np.array([[1] for i in range(self.nsamples)],dtype=float))
        print('scaled feature matrix with intercept: ',np.append(np.array([[1] for i in range(n_samples)],dtype=float),featureMatrix,1))
        return np.append(np.array([[1] for i in range(n_samples)],dtype=float),featureMatrix,1)
      
    def __scaleFeatures(self,npmatrix):
        #print('pre scaled matrix: ',npmatrix)
        #npmatrix=np.apply_along_axis(lambda x: (x-np.mean(x))/(np.std(x)),0,npmatrix)
        #scale to 0-1
        npmatrix=np.apply_along_axis(lambda x: x/max(x),0,npmatrix)
        #npmatrix=np.apply_along_axis(lambda x: x,0,npmatrix)
        print('scaled matrix:')
        print(npmatrix)
        return npmatrix
       
    def __derivativeWRTTheta(self,xi,yi,thetai,i):
        #print('derivative:')
        #print('xi*(yi-h0(theta*xi)) equals: ',xi[i],'*(',yi,'-',self.__sigmoid(np.dot(xi,thetai.transpose())),')')
        #print(xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose()))))
        return xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose())))
  
        
        #this never needs to be accessed ouside of this class so is private 
    def __gradientDescent(self,featureMatrix):
        print('\n\n gradientDescent')
        import pandas as pd
        import numpy as np
        #weights=np.array(initialWeights)
        #print(weights)
        #print(featureMatrix)
        #condition is that total gradient is greater than tolerance
        self.predictors=self.__scaleFeatures(self.predictors)
        self.predictors=self.__addIntercept(self.predictors)
        condition=True
        iterations=0
        while(condition and iterations<self.maxiter):
            gradSquareSum=0 
            print('\n\n\n\n starting weights: ',self.parameterVector)
            print('predictions: ', self.predictOutcome(self.predictors,self.parameterVector))
            tmp=np.copy(self.parameterVector)
            newparams=[]            
            for weight in range(len(self.parameterVector)):
                #compute the derivative
                print('\n\nweight',weight)
                s=0
                print('encoded target:',self.encodedTarget)
                for i in range(len(self.predictors)):
                    #print('i',i)
                    #print('der',self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp))
                    print('working on row: ',i, self.predictors[i], 'yi',self.encodedTarget[i])
                    s=s+self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp,weight)
                    
               
                print('total derivative',s)
                derivative=(-1/len(self.dataframe))*s                
                print(' averaged derivative for theta ',weight)
                print(derivative)
                #print('here',newparams[weight])
                newparams.append(tmp[weight]-self.learningRate*derivative)
                print('newparams[weight]',newparams[weight])
                print('newparams',newparams)
                gradSquareSum+=derivative**2
            print('gradSquareSum: ',gradSquareSum)
            self.parameterVector=np.copy(newparams)
            if(gradSquareSum<self.tolerance):
                condition=False
            iterations+=1
        return 
     #This should only be called once the weights vector has been optimised
    def predictOutcome(self,feature, weights):
        import numpy as np
        print('feature',feature)
        print('weights',weights)
        predictions=list(map(lambda x: self.__sigmoid(x),np.dot(feature,weights.transpose())))
        #print('predictions',predictions)    
        return predictions
    
    def predict_proba(self,x):
        print('pre x matrix',x)
        x=np.matrix(x,dtype=float)
        x=self.__scaleFeatures(x)
        print('post scaled',x)
        x=self.__addIntercept(x)
        print('x',x)
        return list(map(lambda x: self.__sigmoid(x),np.dot(x,self.parameterVector.transpose())))
    
    def predict_class(self,x):
        return list(map(lambda x: 1 if x>0.5 else 0, self.predict_proba(x)))
        
import pandas as pd 
import numpy as np
#dframe=pd.read_csv('/home/user15/Downloads/David/owls15.csv')
#logReg=LogisticRegression(dframe.columns[:-1],dframe.columns[-1], dframe)
#logReg.fit()
        
l1=[[1,1,0],[2,1,0],[1,2,0],[4,5,1],[4,4,1],[5,4,1]]
arr=np.array(l1)
arr2=np.array([[1],[1],[1],[1]])

df = pd.DataFrame(l1,columns=list('ABC'))
logReg=LogisticRegression(df.columns[:-1],df.columns[-1], df)
logReg.trainTestSplit(0.8)
logReg.fit()

print('probability predictions for whole dataset: ',logReg.predict_proba(df[df.columns[:-1]]))
print('class predictions for whole dataset: ',logReg.predict_class(df[df.columns[:-1]]))
test=[[4,4],[5,10]]
#print(logReg.predict_class(pd.DataFrame(test,columns=list('AB'))))
#logReg.trainTestSplit(0.7)

#This gives the first row of the dataframe
#df[0:1]
#This gives the second row of the dataframe
#df[1:2]
#This gives the second row of the dataframe subsetted to first to third columns
#df[1:2][df.columns[1:3]]       
        

#all_data = np.append(arr2,arr, 1)
#l1=c(1,2,1,4,4,5)
#l2=c(1,1,2,5,4,4)
#l3=c(0,0,0,1,1,1)

#dframe=data.frame(l1,l2,l3)
#model <- glm(l3 ~.,family=binomial(link='logit'),data=dframe)
#summary(model)
#model
       
