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
#3).Implement multiclass 1 vs. all predictions  y
#4).Figure how to recover the unscaled weights
#5).Rename everything to train/test to be clear what's what
#6).Think about returning log odds
#7).Give a way for users to enter non-linear combinations of the data
#8).Make a scale_factors array which keeps each scale factor for each column


	

#Numpy matrices are strictly 2-dimensional, while numpy arrays (ndarrays) are N-dimensional. Matrix objects are a subclass of ndarray, so they inherit all the attributes and methods of ndarrays.

#The main advantage of numpy matrices is that they provide a convenient notation for matrix multiplication: if a and b are matrices, then a*b is their matrix product.


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
            self.__preprocess(predictors, target, dataframe)
        #initally set the whole of the 
        self.TrainTestRatio=1
        
        #self.parameterVector=[0 for i in range(len(predictors))]
        self.tolerance=0.00002
        self.learningRate=0.8
        self.maxiter=10
      
    #have multiple implementations of data preprocessing code, do it all here  
    def __preprocess(self, predictors,target,dataframe):
        self.targetName=target
        #this might not be necessary
        self.predictorNames=predictors
        #append on a column of ones
        #dataframe['intercept']=1
        self.predictors=np.matrix(dataframe[predictors],dtype=float)
        self.nsamples=np.shape(self.predictors)[0]
        #add intercept to the predictor features
        self.predictors=self.__addIntercept(self.predictors)
        self.npredictors=np.shape(self.predictors)[1]
        print('predictor shape')
        print(self.predictors.shape)
        self.scaleFactor=self.predictors.max()
        print('self.scaleFactor:')
        print(self.scaleFactor)
        self.target=np.array(dataframe[target])
        self.targetLevels=dataframe[target].unique()
        if len(self.targetLevels)<2:
            print('The target variable has less than 2 unique values and cannot be classified!')
        else:
            if len(self.targetLevels)>=2:
                #each column represents the weights of a model, number of rows gives the number of weights in each model
                self.weightMatrix=np.zeros((self.predictors.shape[1],len(self.targetLevels)))
                #self.weightMatrix[0]=[i for i in range(len(self.targetLevels))]
                #print([len(self.targetLevels)])
                print('self.weightMatrix')
                print(self.weightMatrix)
            #else:
            #    self.weightMatrix=np.zeros((self.predictors.shape[1],1))
            #    print([len(self.targetLevels)])
            #    print('self.weightMatrix')
            #    print(self.weightMatrix)
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

        #Validations complete

    
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
    #takes in single xi value
    def __derivativeWRTTheta(self,xi,yi,thetai,i):
        #print('derivative:')
        #print('xi*(yi-h0(theta*xi)) equals: ',xi[i],'*(',yi,'-',self.__sigmoid(np.dot(xi,thetai.transpose())),')')
        #print(xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose()))))
        return xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose())))
        
    #fit should have 0 or 1 as the target ALWAYS and should probably take a training set to fit
    def fit(self):
        '''The fit method fits n models to the data using a 1 vs. all strategy, where n represents the unique number of levels of the 
        target variable'''
        print('\n\n FIT')
        print('Fitting Logistic Regression Model to ',self.TrainTestRatio*100,'% of the dataset')
        print('length of predictors: ',len(self.predictors))
        print(self.predictors,self.encodedTarget)
        print('number of models equals', self.weightMatrix.shape[1])
        import time
        st=time.clock()
        #apply gradient descent using each set of weights in weight matrix
        print('columns:')
        for columnNo in range(len(self.weightMatrix.transpose())):
            print(self.weightMatrix.transpose()[columnNo])
            print('One vs. all:',self.targetLevels[columnNo])
            print(self.encodedTarget)
            print(self.targetLevels[columnNo])
            oneVSAllTarget=np.array(list(map(lambda x:1 if x==self.targetLevels[columnNo] else 0,self.encodedTarget)))
            print('oneVSAllTarget',oneVSAllTarget)
            self.weightMatrix.transpose()[columnNo]=self.__gradientDescent(self.weightMatrix.transpose()[columnNo],self.predictors,oneVSAllTarget)
        #self.weightMatrix=np.apply_along_axis(self.__gradientDescent,0,self.weightMatrix,self.predictors,self.encodedTarget)
        #self.__gradientDescent(self.predictors)
        print('Fit in ',time.clock()-st, ' seconds')
        print('gradient descent results: ')
        print(self.weightMatrix)
        
        
    #this never needs to be accessed ouside of this class so is private 
    #Given a feature matrix and a weight vector and a target column, return the optimised weight matrix
    def __gradientDescent(self,weightVector,featureMatrix, targetVector):
        print('\n\n gradientDescent')
        print('One vs. all:')
        print(targetVector)
        import pandas as pd
        import numpy as np
        weights=np.array(weightVector)
        featureMatrix=self.__scaleFeatures(featureMatrix)
        #condition initally starts as true
        condition=True
        iterations=0
        while(condition and iterations<self.maxiter):
            #implement one vs all here
            gradSquareSum=0 
            print('\n\n\n\n starting weights: ',weights)
            print('predictions: ', self.predictOutcome(featureMatrix,weights))
            print('len of feature matrix', len(featureMatrix))
            tmp=np.copy(weights)
            newparams=[]            
            for weight in range(len(weights)):
                #compute the derivative
                print('\n\nweight',weight)
                s=0
                print('encoded target:',targetVector)
                for i in range(len(featureMatrix)):
                    #print('i',i)
                    #print('der',self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp))
                    print('working on row: ',i, featureMatrix[i], 'yi',targetVector[i])
                    s=s+self.__derivativeWRTTheta(featureMatrix[i],targetVector[i],tmp,weight)                                  
                print('total derivative',s)
                derivative=(-1/featureMatrix.shape[0])*s                
                print(' averaged derivative for theta ',weight)
                print(derivative)
                #print('here',newparams[weight])
                newparams.append(tmp[weight]-self.learningRate*derivative)
                print('newparams[weight]',newparams[weight])
                print('newparams',newparams)
                gradSquareSum+=derivative**2
            print('gradSquareSum: ',gradSquareSum)
            weights=np.copy(newparams)
            if(gradSquareSum<self.tolerance):
                condition=False
            iterations+=1
        return weights
  
        
    #this never needs to be accessed ouside of this class so is private 
    #Given a feature matrix and a weight matrix and a target column, return the optimised weight matrix
    #def __gradientDescent(self,featureMatrix,weightMatrix, targetVector):
#        print('
#
# gradientDescent')
#        import pandas as pd
#        import numpy as np
#        #weights=np.array(initialWeights)
#        #print(weights)
#        #print(featureMatrix)
#        #condition is that total gradient is greater than tolerance
#        self.predictors=self.__scaleFeatures(self.predictors)
#        #self.predictors=self.__addIntercept(self.predictors)
#        condition=True
#        iterations=0
#        while(condition and iterations<self.maxiter):
#            gradSquareSum=0 
#            print('
#
#
#
# starting weights: ',self.parameterVector)
#            print('predictions: ', self.predictOutcome(self.predictors,self.parameterVector))
#            tmp=np.copy(self.parameterVector)
#            newparams=[]            
#            for weight in range(len(self.parameterVector)):
#                #compute the derivative
#                print('
#
#weight',weight)
#                s=0
#                print('encoded target:',self.encodedTarget)
#                for i in range(len(self.predictors)):
#                    #print('i',i)
#                    #print('der',self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp))
#                    print('working on row: ',i, self.predictors[i], 'yi',self.encodedTarget[i])
#                    s=s+self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp,weight)
#
#
#                print('total derivative',s)
#                derivative=(-1/len(self.dataframe))*s                
#                print(' averaged derivative for theta ',weight)
#                print(derivative)
#                #print('here',newparams[weight])
#                newparams.append(tmp[weight]-self.learningRate*derivative)
#                print('newparams[weight]',newparams[weight])
#                print('newparams',newparams)
#                gradSquareSum+=derivative**2
#            print('gradSquareSum: ',gradSquareSum)
#            self.parameterVector=np.copy(newparams)
#            if(gradSquareSum<self.tolerance):
#                condition=False
#            iterations+=1
#        return 
#     #This should only be called once the weights vector has been optimised
    def predictOutcome(self,feature, weights):
        print('\n\n PREDICT OUTCOME')
        import numpy as np
        print('feature',np.array(feature))
        print('weights',weights)
        print('dot product',np.dot(feature,weights.transpose()))
        predictions=list(map(lambda x: self.__sigmoid(x),np.dot(np.array(feature),weights.transpose())))
        #print('predictions',predictions)    
        return predictions

    #this should take in a dataframe, and check that the relevant columns are present with relevant levels, then split into a feature matrix
    #and a target array and then run the probability predictions
    def predict_proba(self,x):
        '''The predict_proba method returns a list of predicted values for each row in the provided x value. 
        The data in x must align with the provided training data'''
        print('pre x matrix',x)
        x=np.matrix(x,dtype=float)
        x=self.__scaleFeatures(x)
        print('post scaled',x)
        x=self.__addIntercept(x)
        print('x',x)
        #if len(self.uniqueEncodedLevels)==2:
           # return list(map(lambda x: self.__sigmoid(x),np.dot(x,self.weightMatrix)))
        #else:
        f=np.vectorize(self.__sigmoid)
        print(np.dot(x,self.weightMatrix))
        result=f(np.dot(x,self.weightMatrix))
        print(result)
            #print(np.apply_along_axis(,1,np.dot(x,self.weightMatrix)))
        return list(np.apply_along_axis(np.argmax,1,result))
        #list(map(lambda x: self.targetName.index(x),
        #return list(map(lambda x: self.__sigmoid(x),np.dot(x,self.weightMatrix)))
    
    def predict_class(self,x):
        return list(map(lambda x: 1 if x>0.5 else 0, self.predict_proba(x)))
        
import pandas as pd 
import numpy as np
#dframe=pd.read_csv('/home/user15/Downloads/David/owls15.csv')
#logReg=LogisticRegression(dframe.columns[:-1],dframe.columns[-1], dframe)
#logReg.fit()
        
#Test 1        
l=[[1,1,'zero'],[2,1,'zero'],[1,2,'zero'],[4,5,'one'],[4,4,'one'],[5,4,'one']]
df = pd.DataFrame(l,columns=list('ABC'))
logReg=LogisticRegression(df.columns[:-1],df.columns[-1], df)
logReg.fit()
logReg.trainTestSplit(0.9)
print('Class predictions:')
logReg.predict_proba(df[df.columns[:-1]])

#passing in a 5x3 dataframe with 3 unique values. Expecting to see 3x3 weightMatrix 

#Test 2      
l1=[[1,1,0],[2,1,0],[1,2,0],[4,5,1],[4,4,1],[5,4,1]]
df1 = pd.DataFrame(l1,columns=list('ABC'))
logReg1=LogisticRegression(df1.columns[:-1],df1.columns[-1], df1)
logReg1.fit()
logReg1.predict_proba(df1[df1.columns[:-1]])
#class predictions should be [0,0,0,1,1,1]
logReg1.predict_class(df1[df1.columns[:-1]])
#passing in a 5x3 dataframe with 2 unique values. Expecting to see 3x1 weightMatrix 

#Test 2
l2=[[1,1,2,0],[2,1,2,0],[1,2,3,0],[4,5,5,1],[4,4,5,1],[7,7,7,2]]
df2=pd.DataFrame(l2,columns=list('ABCD'))
logReg2=LogisticRegression(df2.columns[:-1],df2.columns[-1], df2)
logReg2.fit()
logReg2.predict_proba(df2[df2.columns[:-1]])
#passing in a 5x4 dataframe with 3 unique values. Expecting to see 4x3 weightMatrix 
owls=pd.read_csv('/home/user15/Downloads/owls15.csv')
owls
logReg3=LogisticRegression(owls.columns[:-1],owls.columns[-1],owls)
logReg3.fit()
logReg3.predict_proba(owls[owls.columns[:-1]])
#logReg.trainTestSplit(1)
#logReg.fit()

#print('probability predictions for whole dataset: ',logReg.predict_proba(df[df.columns[:-1]]))
#print('class predictions for whole dataset: ',logReg.predict_class(df[df.columns[:-1]]))
#test=[[4,4],[5,10]]
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
       
       
def commentCode(string):
    returnString=''
    for line in string.split('\n'):
        returnString+='\n#'+line
    print(returnString)
    
def commentPrint(string):
    returnString=''
    for line in string.split('\n'):
        print(line.strip()[0:5])
        if line.strip()[0:5]=='print':
            returnString+='\n#'+line
        else:
            returnString+='\n'+line
    print(returnString)
       

       
       
