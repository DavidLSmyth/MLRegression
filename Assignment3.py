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

#%%
class LogisticRegression():
    '''This logistic regression class was written as an exercise. The class allows the user to 
    initialise the algorithm with certain values. Call the fit function to fit a set of weights to
    a logistic regression model. The class is initialised by providing the column names of the predictors in the 
    dataframe, the target column name and a pandas dataframe. The learning rate and regularizationValue can 
    also optionally be set'''
    def __init__(self, predictors, target, dataframe,learningRate=2,regularizationValue=0.05):
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
        #initally set the whole of the dataset for training
        self.TrainTestRatio=1
        
        #The names and levels of things do not change
        self.targetName=target
        self.predictorNames=predictors
        #self.parameterVector=[0 for i in range(len(predictors))]
        self.target=np.array(dataframe[target])
        self.targetLevels=dataframe[target].unique()
        #print('unique levels:')
        #print(self.targetLevels)
        self.uniqueEncodedLevels=[i for i in range(len(dataframe[target].unique()))]
        #print('encodedLevels')
        #print(self.uniqueEncodedLevels)
        self.dataframe=dataframe
        self.__preprocessTrain(self.TrainTestRatio)
        #could make these private using __
        self.tolerance=0.00002
        self.learningRate=learningRate
        self.lambdaValue=regularizationValue
        self.maxiter=5000
      
    #have multiple implementations of data preprocessing code, do it all here. Shouldn't need to pass in any params 
    def __preprocessTrain(self,ratio):
        import numpy as np
        randoms=np.random.uniform(0,1,len(self.dataframe))
        print('Random Split: ',randoms)
        self.trainSet=self.dataframe.iloc[randoms<ratio].copy(deep=True)
        
        self.predictors=np.matrix(self.trainSet[self.predictorNames],dtype=float)
        
        self.nsamples=np.shape(self.predictors)[0]
        #add intercept to the predictor features
        self.predictors=self.__addIntercept(self.predictors)
        
        self.npredictors=np.shape(self.predictors)[1]
            
        self.scaleFactor=self.predictors.max()
        if len(self.dataframe.iloc[randoms>=ratio].copy(deep=True)!=0):
            self.testSet=self.dataframe.iloc[randoms>=ratio].copy(deep=True) 
            self.testPredictors=np.matrix(self.testSet[self.predictorNames],dtype=float)
            self.testnsamples=np.shape(self.testPredictors)[0]
            #self.testPredictors=self.__addIntercept(self.testPredictors)
            self.testNPredictors=np.shape(self.predictors)[1]
            self.encodedTargetTest=np.array(self.testSet[self.targetName])
        
        print('self.scaleFactor:')
        print(self.scaleFactor)
        
        self.encodedTarget=np.array(self.trainSet[self.targetName])

        
        if len(self.targetLevels)<2:
            print('The target variable has less than 2 unique values and cannot be classified!')
        else:
            if len(self.targetLevels)>=2:
                #each column represents the weights of a model, number of rows gives the number of weights in each model
                self.weightMatrix=np.zeros((self.predictors.shape[1],len(self.targetLevels)))
                print('self.weightMatrix')
                print(self.weightMatrix)
        #self.encodedTarget=list(map(lambda x: self.uniqueEncodedLevels[list(self.targetLevels).index(x)] ,self.target))
        #print('encodedTarget',self.encodedTarget)
        #self.parameterVector=np.array([0 for i in range(self.predictors.shape[1]+1)])
        self.scaleFactor=self.predictors.max()
        #Validations complete

    
    def trainTestSplit(self, ratio):
        '''Ratio is the desired ratio for the train set size to test set size, no return value'''
        #print('\n\n trainTestSplit')
        self.TrainTestRatio=ratio
        self.__preprocessTrain(ratio)
        print('Data succcessfully split into: ',len(self.trainSet),' training samples and ',len(self.testSet),' testing samples')
        

    def __sigmoid(self,z):
        from math import exp
        #print('z',z)
        return 1/(1+exp(-z))
        
    def __addIntercept(self,featureMatrix):
        n_samples=np.shape(featureMatrix)[0]
        #print(np.array([[1] for i in range(self.nsamples)],dtype=float))
        #print('scaled feature matrix with intercept: ',np.append(np.array([[1] for i in range(n_samples)],dtype=float),featureMatrix,1))
        return np.append(np.array([[1] for i in range(n_samples)],dtype=float),featureMatrix,1)
      
    def __scaleFeatures(self,npmatrix):
        #print('pre scaled matrix: ',npmatrix)
        #npmatrix=np.apply_along_axis(lambda x: (x-np.mean(x))/(np.std(x)),0,npmatrix)
        #scale to 0-1
        npmatrix=np.apply_along_axis(lambda x: x/max(x),0,npmatrix)
        #npmatrix=np.apply_along_axis(lambda x: x,0,npmatrix)
        #print('scaled matrix:',npmatrix)
        return npmatrix
    #takes in single xi value
    def __derivativeWRTTheta(self,xi,yi,thetai,i):
        #print('derivative:')
        #print('xi*(yi-h0(theta*xi)) equals: ',xi[i],'*(',yi,'-',self.__sigmoid(np.dot(xi,thetai.transpose())),')')
        #print(xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose()))))
        return xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose())))
        
    #fit should have 0 or 1 as the target ALWAYS and should probably take a training set to fit
    def fit(self,columnNames=[]):
        '''The fit method fits n models to the data using a 1 vs. all strategy, where n represents the unique number of levels of the 
        target variable'''
        
        #here subset some of the columns for the user to specify only certain columns
        if(columnNames!=[]):
            pass
        print('\n\n FIT')
        print('Fitting Logistic Regression Model to ',self.TrainTestRatio*100,'% of the dataset')
        print('length of predictors: ',len(self.predictors))
        #print(self.predictors,self.encodedTarget)
        print('number of models equals', self.weightMatrix.shape[1])
        import time
        st=time.clock()
        #apply gradient descent using each set of weights in weight matrix
        #print('columns:')
        for columnNo in range(len(self.weightMatrix.transpose())):
            #print(self.weightMatrix.transpose()[columnNo])
            #print('One vs. all:',self.targetLevels[columnNo])
            #print(self.encodedTarget)
            #print(self.targetLevels[columnNo])
            oneVSAllTarget=np.array(list(map(lambda x:1 if x==self.targetLevels[columnNo] else 0,self.encodedTarget)))
            #print('oneVSAllTarget',oneVSAllTarget)
            self.weightMatrix.transpose()[columnNo]=self.__gradientDescent(self.weightMatrix.transpose()[columnNo],self.predictors,oneVSAllTarget)
        #self.weightMatrix=np.apply_along_axis(self.__gradientDescent,0,self.weightMatrix,self.predictors,self.encodedTarget)
        #self.__gradientDescent(self.predictors)
        print('Fit in ',time.clock()-st, ' seconds')
        
        #Try and print the results of each model vs. the other models for 1 vs. all
        print('gradient descent results: ')
        print(self.weightMatrix)
        
        
    #this never needs to be accessed ouside of this class so is private 
    #Given a feature matrix and a weight vector and a target column, return the optimised weight matrix
    def __gradientDescent(self,weightVector,featureMatrix, targetVector):
        import time
        start=time.clock()
        print('\n\n gradientDescent')
        print('One vs. all:')
        #print(targetVector)
        import pandas as pd
        import numpy as np
        weights=np.array(weightVector)
        featureMatrix=self.__scaleFeatures(featureMatrix)
        #condition initally starts as true
        condition=True
        iterations=0
        self.gradientMagnitude=[]
        while(condition and iterations<self.maxiter):
            #implement one vs all here
            gradSquareSum=0 
            #print('\n\n\n\n starting weights: ',weights)
            #print('predictions: ', self.predictOutcome(featureMatrix,weights))
            #print('len of feature matrix', len(featureMatrix))
            tmp=np.copy(weights)
            newparams=[]            
            for weight in range(len(weights)):
                #compute the derivative
                #print('\n\nweight',weight)
                s=0
                #print('encoded target:',targetVector)
                for i in range(len(featureMatrix)):
                    #print('i',i)
                    #print('der',self.__derivativeWRTTheta(self.predictors[i],self.encodedTarget[i],tmp))
                    #print('working on row: ',i, featureMatrix[i], 'yi',targetVector[i])
                
                    #have an np.apply_to_colomn call here
                    s=s+self.__derivativeWRTTheta(featureMatrix[i],targetVector[i],tmp,weight)
                #print('total derivative',s)
                derivative=(-1/featureMatrix.shape[0])*s                
                #print(' averaged derivative for theta ',weight)
                #print(derivative)
                #print('here',newparams[weight])
                newparams.append(tmp[weight]*(1-((self.learningRate*self.lambdaValue)/featureMatrix.shape[0]))-self.learningRate*derivative)
                #print('newparams[weight]',newparams[weight])
                #print('newparams',newparams)
                gradSquareSum+=derivative**2
            self.gradientMagnitude.append(derivative**2)   
            #print('gradSquareSum: ',gradSquareSum)
            weights=np.copy(newparams)
            if(gradSquareSum<self.tolerance or (time.clock()-start)>2):
                condition=False
            iterations+=1
        return weights

    #plotConvergence function is provided so that the user can check if the algorithm has converged to a solution if there 
    #are any issues with prediction
    def plotConvergence(self):
        #check here whether self.gradientMagnitude exists
        import matplotlib.pyplot as plt
        print('len gradientMagnitude: ',len(self.gradientMagnitude))
        plt.plot([i for i in range(len(self.gradientMagnitude))],self.gradientMagnitude)
        plt.xlabel('Iteration number')
        plt.ylabel('Magnitude of derivative')
        plt.title('Plot of gradient magnitude vs. iteration')
        
    def predict_test(self):
        if len(self.testSet)>0:
            #check that the test set contains the same number of levels as the train set
            #print('self.testPredictors[self.targetName].unique()')
            #print(self.encodedTargetTest)
            #print(list(np.unique(self.encodedTargetTest)))
            #print(sorted(self.targetLevels))
            #print(sorted(np.unique(self.encodedTargetTest)))
            if sorted(self.targetLevels) == sorted(np.unique(self.encodedTargetTest)):
                testResults=self.predict_class(self.testPredictors)
                print('\n\n\n\n')
                print(list(zip(testResults,self.encodedTargetTest)))
                print('\n\n\n')
                accuracy=sum(list(map(lambda x: x[0]==x[1],zip(testResults,self.encodedTargetTest))))/len(testResults)
                print('Test Accuracy',accuracy)
                return accuracy
                


    def assignmentMethod(self):
        accuracies=[]
        for i in range(10):
            self.trainTestSplit(2/3)
            self.fit()
            accuracies.append(self.predict_test())
        print('accuracies',accuracies)
    #this should take in a dataframe, and check that the relevant columns are present with relevant levels, then split into a feature matrix
    #and a target array and then run the probability predictions
    def predict_class(self,x):
        '''The predict_proba method returns a list of predicted values for each row in the provided x value. 
        The data in x must align with the provided training data'''
        #print('pre x matrix',x)
        x=np.matrix(x,dtype=float)
        x=self.__scaleFeatures(x)
        #print('post scaled',x)
        x=self.__addIntercept(x)
        #print('x with intercept',x)
        #print('x',x)
        f=np.vectorize(self.__sigmoid)
        #print(np.dot(x,self.weightMatrix))
        result=f(np.dot(x,self.weightMatrix))
        #print(result)
            #print(np.apply_along_axis(,1,np.dot(x,self.weightMatrix)))
        #print(list(np.apply_along_axis(np.argmax,1,result)))
        #print(list(self.targetLevels))
        #print(list(map(lambda x: list(self.targetLevels)[x],list(np.apply_along_axis(np.argmax,1,result)))))
        return list(map(lambda x: list(self.targetLevels)[x],list(np.apply_along_axis(np.argmax,1,result))))
        #list(map(lambda x: self.targetName.index(x),
        #return list(map(lambda x: self.__sigmoid(x),np.dot(x,self.weightMatrix)))
    
    #def predict_class(self,x):
     #   return list(map(lambda x: 1 if x>0.5 else 0, self.predict_proba(x)))

#%%     
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
logReg.predict_class(df[df.columns[:-1]])
#%%
#passing in a 5x3 dataframe with 3 unique values. Expecting to see 3x3 weightMatrix 

#Test 2      

l1=[[1,1,0],[2,1,0],[1,2,0],[4,5,1],[4,4,1],[5,4,1]]
df1 = pd.DataFrame(l1,columns=list('ABC'))
logReg1=LogisticRegression(df1.columns[:-1],df1.columns[-1], df1)
logReg1.fit()
logReg1.trainTestSplit(2/3)
logReg1.predict_class(df1[df1.columns[:-1]])
logReg1.predict_test()
#class predictions should be [0,0,0,1,1,1]
#passing in a 5x3 dataframe with 2 unique values. Expecting to see 3x1 weightMatrix 
#%%
#Test 2
l2=[[1,1,2,0],[2,1,2,0],[1,2,3,0],[4,5,5,1],[4,4,5,1],[7,7,7,2]]
df2=pd.DataFrame(l2,columns=list('ABCD'))
logReg2=LogisticRegression(df2.columns[:-1],df2.columns[-1], df2)
logReg2.fit()
logReg2.plotConvergence()
logReg2.predict_class(df2[df2.columns[:-1]])
#passing in a 5x4 dataframe with 3 unique values. Expecting to see 4x3 weightMatrix 
#%%
import pandas as pd 
import numpy as np
owls=pd.read_csv('/home/user15/Downloads/owls15.csv',names=['body-length', 'wing-length', 'body-width', 'wing-width','type'])
owls =  owls.reindex(np.random.permutation(owls.index))
logReg3=LogisticRegression(owls.columns[:-1],owls.columns[-1],owls)
logReg3.assignmentMethod()
logReg3.trainTestSplit(2/3)
logReg3.fit()
logReg3.plotConvergence()
predictions=logReg3.predict_class(owls[owls.columns[:-1]])
owls['predictions']=predictions
print(sum(owls['type']==owls['predictions']))
logReg3.predict_test()


#%%

uniqueLevels=list(owls['LongEaredOwl'].unique())
list(map(lambda x: uniqueLevels[x],predictions))
print(owls.columns)
owls['predictions']=predictions
print(sum(owls['type']==owls['predictions']))
owls=owls.replace('BarnOwl','SnowyOwl')
print(owls[owls.iloc('type')])
print(len(owls))

owls['type'].unique()
#plot long-eared vs.all
owls1=owls.replace('BarnOwl','SnowyOwl')
#plot barn vs. all
owls2=owls.replace('SnowyOwl','LongEaredOwl')
#plot snowy vs. all
owls3=owls.replace('LongEaredOwl','BarnOwl')

owls4=owls.replace('BarnOwl','SnowyOwl')

pythonUnivar(owls1,'body-length','type','LongEaredOwl',bins=15)
pythonUnivar(owls1,'wing-length','type','LongEaredOwl',bins=15)
pythonUnivar(owls1,'body-width','type','LongEaredOwl',bins=15)
pythonUnivar(owls1,'wing-width','type','LongEaredOwl',bins=15)

#[[-4.4714721   2.15491399  1.25347074]
# [-1.92739136 -5.45135104  3.68106958]
# [-1.38819945  0.49770969 -0.30225073]
# [ 3.09855626  2.04099072 -4.68412034]
# [ 5.50350867  0.03109874 -5.62315064]]

#%%
def pythonUnivar(dataframe,feature, target, targetSuccessName=None, prediction=None,bins=10): 
    #dataframe should be a pandas dataframe
    #feature, target, prediction are strings
    #bins is an integer
    #add errorbars
    '''pythonUnivar is a function that will plot a histogram which has:
    
    1). n bins on the x-axis which partition the data set according to the provided feature
    2). A count on the y axis of the amount of people in each bin
    3). A line plot of how the target variable '1' value is distributed across each of the bins
    4). Optional: If there is a prediction column in the data set, a A line plot of how the predicted values are distributed across each of the bins
        
    args: 
    Dataframe is a pandas dataframe
    Feature is a column name of the dataframe by which the data will be partitioned
    Target is the dependent variable
    Prediction is a column of predictions of the dependent variable
    Bins is the number of bins to partition the data into, based on the distribution of the dependent variable.
        
    Example Call:
    '''
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    #==============================================================================
    #     implement error checking here    
    #==============================================================================
    #check that feature exists in dataframe, if not give suggestions 
    #check data can actually be divided up into given number of bins
    #check that prediction is also a binary variable

    #check validity of data, code is self documenting
    if(type(dataframe)!=pd.core.frame.DataFrame):
        print('You must provide a pandas dataframe, the dataframe you have provided is of type '+type(dataframe))
        return
    elif feature not in dataframe.columns:
        print(feature+' is not a valid column name')
        return 
    elif target not in dataframe.columns:
        print(target+' is not a valid column name')
        return
    elif ((prediction!=None) and (prediction not in dataframe.columns)):
        print(prediction+' is not a valid column name')
        return
    elif len(dataframe[target].unique())!=2:
        print('The target variable must be binary for this plot!')
        print(target+' has '+str(len(dataframe[target].unique()))+' columns')
        return
    elif targetSuccessName not in dataframe[target].unique() and targetSuccessName!=None:
        print(targetSuccessName+' is not present in the target column')
    else:
        #encode the target variable to 1-0 binary values
        targetVar=dataframe[target]
        targetVals=dataframe[target].unique()
        if targetSuccessName==None:
            targetSuccessName=targetVals[0]
 
        print('targetVals: 1',targetVals[0])
        dataframe[target]=np.where(dataframe[target]==targetSuccessName, 1, 0) 
        print(dataframe[target])
        #create a new matplotlib figure
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        #add title
        plt.title('Histogram of '+feature+' and proportion of '+str(targetVals[0])+' in each bin')
        #add second axis to give proporion of 1 values per bin
        ax2 = ax1.twinx()
        #label both axes
        ax2.set_ylabel('Proportion of '+str(targetVals[0])+' per bin')
        ax1.set_ylabel('Count of '+str(targetVals[0])+' in each bin')
        #create a regular histogram
        n=ax1.hist(dataframe[feature], bins=bins, normed=False, alpha=0.5)
        for i in n:
            print(i)
        #print(binedges)
        #ya,binedges=np.histogram(dataframe[feature], bins=bins, density=True)
        #ax1.plot(ya,binedges,color='green')
        #ax1.hist(dataframe[feature], density=True)
        #now add the line plot which shows how the response varies
        y,binEdges=np.histogram(dataframe[feature],weights=dataframe[target], bins=bins)
        print(binEdges)
        print('y')
        print(y)
        print('y/n[0]')
        print(np.nan_to_num((y/n[0])))
        bincenters = (0.5*(binEdges[1:]+binEdges[:-1]))
        #mother of god this took ages
        ax1.plot(bincenters,(ax1.get_yticks()[-1])*(np.nan_to_num((y/n[0]))),'-', color='red', label='Target value density')
        
        
    #if there is a prediction, then also add it to the plot to see how closely it corresponds
    #to the target variable
    if prediction!=None:
        y2,binEdges2=np.histogram(dataframe[feature],weights=dataframe[prediction], bins=bins)
        bincenters2=0.5*(binEdges2[1:]+binEdges2[:-1])
        ax1.plot(bincenters2,(ax1.get_yticks()[-1])*(y2/n[0]),'-', color='green',label='Prediction density')
        print(y2)
        print(n[0])
        print((ax1.get_yticks()[-1])*(y2/n[0]))
    #replace the target variable with original non=numeric values
    ax1.legend(loc='upper left')
    dataframe[target]=targetVar
    return



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
#%%     
       
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
       

       
       
