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
#3).Option to return theta values, std. errors & p values would also be nice

#Try plotting non-convex J(theta) cost function for better understanding

#TODO:
#1).tidy up current class  y
#2).Vectorise current implementation  y
#3).Implement multiclass 1 vs. all predictions  y
#4).Rename everything to train/test to be clear what's what  y - if not named test assume train
#5).Give a way for users to enter non-linear combinations of the data
#6).Make a scale_factors array which keeps each scale factor for each column  y

#Difference between numpy matrices and numpy arrays:
#Numpy matrices are strictly 2-dimensional, while numpy arrays (ndarrays) are N-dimensional.
# Matrix objects are a subclass of ndarray, so they inherit all the attributes and methods of ndarrays.
#The main advantage of numpy matrices is that they provide a convenient notation for matrix 
#multiplication: if a and b are matrices, then a*b is their matrix product.

#%%
class LogisticRegression():
    '''This logistic regression class was written as an exercise. The class allows the user to 
    initialise the algorithm with certain values. Call the fit function to fit a set of weights to
    a logistic regression model. The class is initialised by providing the column names of the predictors in the 
    dataframe, the target column name and a pandas dataframe. The learning rate and regularizationValue can 
    also optionally be set'''
    def __init__(self, predictors, target, dataframe,learningRate=1,regularizationValue=0.1):
        import pandas as pd
        import numpy as np
        '''Pass in a list of predictors as strings, a target as a string, a pandas dataframe, optional: learning rate and regularization value'''
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
        elif len(dataframe[target].unique())>=len(dataframe[target])*0.8:
            print('Logistic regression may not be suited to this problem, there are ',len(dataframe['target'].unique()),
            ' predictors which indicated this may not be a classification task')                                                                           
        else:
            #initally set the whole of the dataset for training
            self.TrainTestRatio=1
            #The names and levels of things do not change can be set here, everything else is passed to preprocess
            self.targetName=target
            self.predictorNames=predictors
            self.target=np.array(dataframe[target])
            self.targetLevels=dataframe[target].unique()
            self.uniqueEncodedLevels=[i for i in range(len(dataframe[target].unique()))]
            self.dataframe=dataframe
            #now preprocess the data that can change once the class has been initialised
            self.__preprocessTrain(self.TrainTestRatio)
            #could make these private using __
            self.tolerance=0.0000002
            self.learningRate=learningRate
            self.lambdaValue=regularizationValue
            self.maxiter=8000
      
    #Do all data preprocessing here rather than having multiple implementations 
    def __preprocessTrain(self,ratio):
        import numpy as np
        #randomly shuffle the data
        newNos=np.random.permutation(self.dataframe.index)
        #reindex the dataframe and divide into train and test set
        self.trainSet=self.dataframe.reindex(newNos).head(np.round(len(self.dataframe)*ratio))
        self.testSet=self.dataframe.reindex(newNos).tail(np.round(len(self.dataframe)*(1-ratio)))
        #create a train predictor matrix
        self.predictors=np.matrix(self.trainSet[self.predictorNames],dtype=float)
        #add intercept to the predictor features
        self.predictors=self.__addIntercept(self.predictors)
        #create a scaleFactor vector to record how much each column has been scaled by
        self.scaleFactor=self.predictors.max()
        #It's useful to have a separate target vector
        self.encodedTarget=np.array(self.trainSet[self.targetName])
        #if the test set has at least one value, then split it up in a similar way to the train set
        if len(self.trainSet)!=len(self.dataframe):
            self.testPredictors=np.matrix(self.testSet[self.predictorNames],dtype=float)
            self.encodedTargetTest=np.array(self.testSet[self.targetName])
        if len(self.targetLevels)<2:
            print('The target variable has less than 2 unique values and cannot be classified!')
        else:
            if len(self.targetLevels)>=2:
                #each column represents the weights of a model, number of rows gives the number of weights in each model
                self.weightMatrix=np.zeros((self.predictors.shape[1],len(self.targetLevels)))

    
    def trainTestSplit(self, ratio):
        '''Ratio is the desired ratio for the train set size to test set size, no return value'''
        self.TrainTestRatio=ratio
        self.__preprocessTrain(ratio)
        self.fit()
        print('Data succcessfully split into: ',len(self.trainSet),' training samples and ',len(self.testSet),' testing samples')
    
    #simply adds an intercept to a featureMatrix
    def __addIntercept(self,featureMatrix):
        return np.append(np.array([[1] for i in range(np.shape(featureMatrix)[0])],dtype=float),featureMatrix,1)
      
    #scales the features of a featureMatrix to speed up convergence of gradient descent
    def __scaleFeatures(self,npmatrix):
        return np.apply_along_axis(lambda x: x/max(x),0,npmatrix)
        
    #Redundant since vectorised but leaving in for reference
    def __derivativeWRTTheta(self,xi,yi,thetai,i):
        return xi[i]*(yi-self.__sigmoid(np.dot(xi,thetai.transpose())))
    
    #Again private, hypothesis function predicts the probability of each data point in the feature matrix of being a one based on given weights
    def __sigmoid(self, X, Weight):
        return 1.0/(1+np.exp(- np.dot(X,Weight)))
        
    #fit should have 0 or 1 as the target ALWAYS and should take a training set to fit
    def fit(self,columnNames=[],verbose=True):
        '''The fit method fits n models to the data using a 1 vs. all strategy, where n represents the unique number of levels of the 
        target variable. verbose=False supresses the model summary output.'''
        #here subset some of the columns for the user to specify only certain columns
        if(columnNames!=[]):
            #subset to only the columns that the user provides
            pass
        print('\nFitting Logistic Regression Model to ','%s'%float('%.5g' % (self.TrainTestRatio*100)),'% of the dataset')
        import time
        st=time.clock()
        #apply gradient descent using each set of weights in weight matrix
        for columnNo in range(len(self.weightMatrix.transpose())):
            #Map ith variable to 1 and all other levels in the target vector to 0
            oneVSAllTarget=np.array(list(map(lambda x:1 if x==self.targetLevels[columnNo] else 0,self.encodedTarget)))
            #Update the weights of the ith model (corresponding to a column in self.weightMatrix) using gradient descent
            self.weightMatrix.transpose()[columnNo]=self.__gradientDescent(self.weightMatrix.transpose()[columnNo],self.predictors,oneVSAllTarget)
        print('Fit in ','%s'% float('%.5g'%(time.clock()-st)), ' seconds')
        if verbose:
            self.__fitResults()
       
    def __fitResults(self):
        '''Nicely outputs the results of the model once it has been fit'''
        for columnNo in range(len(self.weightMatrix.transpose())): 
            print('\n---------------------------------------------------')
            print('Coefficients for ',self.targetLevels[columnNo],' vs. all model')
            print('Intercept'+' '*8+' : '+' '*8,'%s'% float('%.5g'%self.weightMatrix[0][columnNo]))                   
            for coef in range(1,len(self.weightMatrix)):
                print(self.predictorNames[coef-1], ' '*(16-len(self.predictorNames[coef-1]))+' : '+' '*8, '%s'% float('%.5g'%self.weightMatrix.item(coef,columnNo)))
        print('---------------------------------------------------')
        
    #this never needs to be accessed ouside of this class so is private 
    #Given a feature matrix and a weight vector and a target column, return the optimised weight matrix
    #featureMatrix is self.predictors
    def __gradientDescent(self,weightVector,featureMatrix, targetVector):
        import time
        import pandas as pd
        import numpy as np
        start=time.clock()
        weights=np.array(weightVector)
        #Scale features to speed up gradient descent
        featureMatrix=self.__scaleFeatures(featureMatrix)
        #condition initally starts as true, gives a do-while effect
        condition=True
        iterations=0
        #Record gradientMagnitude to plot convergence
        self.gradientMagnitude=[]
        #No do-while structure in python, but while condition, and then update condition is a good substitute
        while(condition and iterations<self.maxiter):
            weightsCopy=np.copy(weights)
            #compute derivative of cost function
            s1=(1/featureMatrix.shape[0])*(np.dot(featureMatrix.T,(self.__sigmoid(featureMatrix,weightsCopy))-targetVector))
            #don't regularise intercept            
            s10=s1[0]
            #regularise everything else
            s1+=(weightsCopy*self.lambdaValue)/(featureMatrix.shape[0])
            #reset pre-regularised intercept
            s1[0]=s10
            #update weights based on derivative
            weightsCopy-=self.learningRate*(s1)
            weights=np.copy(weightsCopy)
            gradSquareSum=np.dot(s1,s1.T)
            self.gradientMagnitude.append(gradSquareSum)
            #Gradient descent stops running when:
            #either iterations exceeds max. number of iterations, the gradient becomes acceptable small or 8 seconds elapse.
            if(gradSquareSum<self.tolerance or (time.clock()-start>8)):
                condition=False
            iterations+=1
        #return optimised weights vector
        return weights



    #plotConvergence function is provided so that the user can check if the algorithm has converged to a solution if there 
    #are any issues with prediction
    def plotConvergence(self):
        '''Provides a plot of gradient magnitude vs. iteration number for gradient descent algorihtm.'''
        #check here whether self.gradientMagnitude exists
        import matplotlib.pyplot as plt
        #Plot iteration number vs. gradient magnitude for each iteration
        plt.plot([i for i in range(len(self.gradientMagnitude))],self.gradientMagnitude)
        plt.xlabel('Iteration number')
        plt.ylabel('Magnitude of derivative')
        plt.title('Plot of gradient magnitude vs. iteration')
        
    def predict_test(self,toFile=False):
        #Check that the testSet actually exists!
        if len(self.testSet)>0:
            #check that the test set contains the same number of levels as the train set
            if sorted(self.targetLevels) == sorted(np.unique(self.encodedTargetTest)):
                #testResults contain predictions of the test set predictors
                testResults=self.predict_class(self.testPredictors)
                #Accuracy is the number of predictions that match the actual values divided by the number of datapoints in the test set
                accuracy=sum(list(map(lambda x: x[0]==x[1],zip(testResults,self.encodedTargetTest))))/len(testResults)
                print('\nTest Accuracy','%s'% float('%.4g'%accuracy))
                if toFile:
                    #simple GUI is convenient to use
                    from tkinter.filedialog import asksaveasfile
                    writelocation=asksaveasfile(mode='w+',defaultextension='.txt')
                    writelocation.write('Predicted | Actual \n')
                    #zip predicted an actual into tuples
                    pairedActualPredicted=zip(*[list(testResults),list(self.encodedTargetTest)])
                    for row in pairedActualPredicted:
                        #write each tuple to the file
                        writelocation.write('\n'+str(row))
                        if row[0]!=row[1]:
                            #Note any incorrect predictions
                            writelocation.write(' * incorrect prediction')
                    writelocation.close()
                return accuracy
                
    def TrainTestRandomTen(self,toFile=False):
        '''TrainTestRandomTen is a method specific to the assignment. It randomly splits the dataset 10 times into 
        2/3 train and 1/3 test, where each of the train sets are used to train a model which is then tested against
        their corresponding test sets'''
        accuracies=[]
        for i in range(10):
            #randomly split the data
            self.trainTestSplit(2/3)
            #fit a model to each train set
            self.fit(verbose=False)
            #then append the test set accuracy to an array
            accuracies.append(self.predict_test(toFile))
        print('\nAccuracies',list(map(lambda x: '%s'% float('%.4g'%x),accuracies)))
        print('Average Accuracy: ','%s'% float('%.4g'%np.mean(accuracies)))
        print('Accuracy standard deviation: ','%s'% float('%.4g'%np.std(accuracies)))
        
    def predict_class(self,x,verbose=False):
        '''The predict_class method returns a list of predicted values for each row in the provided x feature. 
        The data in x must align with the provided training data'''
        x=np.matrix(x,dtype=float)
        x=self.__scaleFeatures(x)
        x=self.__addIntercept(x)
        #Non-vectorised implementations needed to vectorise the sigmoid function, keeping for reference
        #f=np.vectorize(self.__sigmoid)
        #result=f(np.dot(x,self.weightMatrix))
        if verbose:
            print('Each column represents the probability of a one vs. all class')
            print(self.targetLevels)
            print(self.__sigmoid(x,self.weightMatrix))
        #result gives the probability predictions of each model
        result=self.__sigmoid(x,self.weightMatrix)
        #Map the highest probability back to it's actual (possibly non-numeric) value
        return list(map(lambda x: list(self.targetLevels)[x],list(np.apply_along_axis(np.argmax,1,result))))

    

#%%     
import pandas as pd 
import numpy as np
#change this to your own downloads folder
owls=pd.read_csv('/home/user15/Downloads/owls15.csv',names=['body-length', 'wing-length', 'body-width', 'wing-width','type'])
owls=owls.reindex(np.random.permutation(owls.index))
logReg3=LogisticRegression(owls.columns[:-1],owls.columns[-1],owls,0.9,0.15)
logReg3.fit()
logReg3.plotConvergence()
import pickle
file_name='testfile'
fileObject=open(file_name,'wb')
pickle.dump(logReg3,fileObject)
fileObject.close()


fileObject=open(file_name,'rb')
deserialized=pickle.load(fileObject)
deserialized.trainTestSplit(0.65)
deserialized.fit()
deserialized.predict_test()


logReg3.trainTestSplit(0.75)
logReg3.predict_test()
logReg3.TrainTestRandomTen()

predictions=logReg3.predict_class(owls[owls.columns[:-1]])
owls['predictions']=predictions
print('Train Accuracy')
print(sum(owls['type']==owls['predictions'])/len(owls))



#%%   
#Test 1        
l=[[1,1,'zero'],[2,1,'zero'],[1,2,'zero'],[4,5,'one'],[4,4,'one'],[5,4,'one']]
df = pd.DataFrame(l,columns=list('ABC'))
logReg=LogisticRegression(df.columns[:-1],df.columns[-1], df)
logReg.fit()
print('Class predictions:')
logReg.predict_class(df[df.columns[:-1]],True)


#%%
#Test 2      

l1=[[1,1,0],[2,1,0],[1,2,0],[4,5,1],[4,4,1],[5,8,1]]
df1 = pd.DataFrame(l1,columns=list('ABC'))
logReg1=LogisticRegression(df1.columns[:-1],df1.columns[-1], df1)
logReg1.fit()
logReg1.trainTestSplit(2/3)
logReg1.predict_class(df1[df1.columns[:-1]])
logReg1.predict_test()
#class predictions should be [0,0,0,1,1,1]
#%%
#Test 3
l2=[[1,1,2,0],[2,1,2,0],[1,2,3,0],[4,5,5,1],[4,4,5,1],[7,7,7,2]]
df2=pd.DataFrame(l2,columns=list('ABCD'))
logReg2=LogisticRegression(df2.columns[:-1],df2.columns[-1], df2)
logReg2.fit()
logReg2.predict_class(df2[df2.columns[:-1]])
#%%
import pandas as pd 
import numpy as np
owls=pd.read_csv('C:\\Users\\Marion\\Downloads\\owls15.csv',names=['body-length', 'wing-length', 'body-width', 'wing-width','type'])
import matplotlib.pyplot as plt
owls.groupby('type')['body-length'].mean()
owls.groupby('type')['wing-length'].mean()
owls.groupby('type')['body-width'].mean()
owls.groupby('type')['wing-width'].mean()

c=[]
c=['red']*len(owls[owls['type']=='LongEaredOwl'])
c+=(['green']*len(owls[owls['type']=='SnowyOwl']))
c+=(['blue']*len(owls[owls['type']=='BarnOwl']))

plt.title('Body length vs. Other variables')
plt.scatter(owls['body-length'],owls['wing-length'],c=c)
plt.scatter(owls['body-length'],owls['body-width'],c=c)
plt.scatter(owls['body-length'],owls['wing-width'],c=c)

plt.title('Scatter plot of each pair of variables superimposed')
plt.scatter(owls['wing-length'],owls['body-length'],c=c)
plt.scatter(owls['wing-length'],owls['body-width'],c=c)
plt.scatter(owls['wing-length'],owls['wing-width'],c=c)

plt.title('Scatter plot of each pair of variables superimposed')
plt.scatter(owls['body-width'],owls['body-length'],c=c)
plt.scatter(owls['body-width'],owls['wing-length'],c=c)
plt.scatter(owls['body-width'],owls['wing-width'],c=c)

plt.title('Scatter plot of each pair of variables superimposed')
plt.scatter(owls['wing-width'],owls['body-length'],c=c)
plt.scatter(owls['wing-width'],owls['wing-length'],c=c)
plt.scatter(owls['wing-width'],owls['body-width'],c=c)

#%%
owls=pd.read_csv('C:\\Users\\Marion\\Downloads\\owls15.csv',names=['body-length', 'wing-length', 'body-width', 'wing-width','type'])
owls=owls.reindex(np.random.permutation(owls.index))
logReg3=LogisticRegression(owls.columns[:-1],owls.columns[-1],owls,0.9,0.15)
logReg3.fit()

logReg3.TrainTestRandomTen()
logReg3.plotConvergence()
predictions=logReg3.predict_class(owls[owls.columns[:-1]])
owls['predictions']=predictions
print(sum(owls['type']==owls['predictions']))
print(owls.loc[owls['type']!=owls['predictions']])
logReg3.predict_test()


#%%
owls['type'].unique()
#plot long-eared vs.all
owls1=owls.replace('BarnOwl','SnowyOwl')
#plot barn vs. all
owls2=owls.replace('SnowyOwl','LongEaredOwl')
#plot snowy vs. all
owls3=owls.replace('LongEaredOwl','BarnOwl')

owls4=owls.replace('BarnOwl','SnowyOwl')

pythonUnivar(owls3,'body-length','type','SnowyOwl',bins=10)
pythonUnivar(owls3,'body-width','type','SnowyOwl',bins=10)
pythonUnivar(owls3,'wing-length','type','SnowyOwl',bins=10)
pythonUnivar(owls3,'wing-width','type','SnowyOwl',bins=10)
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
        plt.title('Histogram of '+feature+' and proportion of '+str(targetSuccessName)+' in each bin')
        #add second axis to give proporion of 1 values per bin
        ax2 = ax1.twinx()
        #label both axes
        ax2.set_ylabel('Proportion of '+str(targetSuccessName)+' per bin')
        ax1.set_ylabel('Count of '+str(targetSuccessName)+' in each bin')
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
       

       
       
