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
#4).Implement a predict method which returns predicted classes (and probabilities if I have time)


#Details of fitting the theta vector: Gradient descent will need to be used, ideally vectorised 
#If I have a n multi-class prediction problem, then I need n models to be fitted for each of the n classes
#Details of prediction: If only two classes, straightforward. If multiclass, then return highest probability
#of results of each of the n models above
#A challenge will be to provided the right level of abstratction

#1). Logistic regression that will ouput probabilities.
#Try plotting non-convex J(theta) cost function for better understanding
class LogisticRegression():
    def __init__(self, predictors, target, dataframe):
        import pandas as pd
        import numpy as np
        from math import exp
        import time
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
        #this might not be necessary
        self.predictorNames=predictors
        self.predictors=np.matrix(dataframe[predictors])
        self.target=np.matrix(dataframe[target])
        self.dataframe=dataframe
        self.parameterVector=np.zeros([1,len(predictors)])
        #self.parameterVector=[0 for i in range(len(predictors))]
        self.tolerance=1**-5
        #Validations complete
        
    def fit(self):
        pass
        
    def __sigmoid(z):
        return 1/(1+exp(-z))
        
    def __derivativeWRTTheta(xi,yi,thetai):
        return xi*(yi-self.__sigmoid(xi*thetai))
        
    #def __gradientDescent(self,featureMatrix,output,initialWeights,stepSize,tolerance):
    #    st=time.clock()    
    #    weights=np.array(initialWeights)
    #    #print(weights)
    #    #print(featureMatrix)
    #    while(True):
    #        gradSquareSum=0 
    #        #time.sleep(0.1)
    #        predictions=predictOutcome(featureMatrix,weights)
    #        errors=predictions-output
    #        print('errors',errors)
    #        print('ave error',(sum(errors)/len(errors)))
    #        tmp=np.array([0.0 for b in range(len(weights))])       
    #        for i in range(len(weights)):
    #            #prediction=predictOutcome(featureMatrix[:, i],weights[i])
    #            derivative=featureDerivative(errors,featureMatrix[:, i])
    #            gradSquareSum=gradSquareSum+derivative**2
    #            #print('gsqsum',gradSquareSum)            
    #            tmp[i]=float(derivative*stepSize)
    #           # print('add weights',float(derivative*stepSize))            
    #            print('der*step',-derivative*stepSize)
    #        print('tmp',tmp)
    #        weights=weights+tmp
    #        gradientmag=np.sqrt(gradSquareSum)
    #        print('gradmag',gradientmag) 
    #        #print('weights',weights)
    #        if gradientmag<tolerance:
    #            print("finished")            
    #            break
    #    print(time.clock()-st)
    #    return weights
        
    #this never needs to be accessed ouside of this class so is private 
    def __gradientDescent(self,featureMatrix,output,initialWeights,stepSize,tolerance):
        st=time.clock()    
        weights=np.array(initialWeights)
        #print(weights)
        #print(featureMatrix)
        
        #condition is that total gradient is greater than tolerance
        condition=True
        while(condition):
            gradSquareSum=0 
            #time.sleep(0.1)
            predictions=predictOutcome(self.predictors,self.parameterVector)
            errors=predictions-self.target
            #print('errors',errors)
            print('ave error',(sum(errors)/len(errors)))
            tmp=np.array([0.0 for b in range(len(weights))])   
            
            for weight in range(len(self.parameterVector)):
                #compute the derivative
                derivative=sum(map(lambda x,y,theta:self.__derivativeWRTTheta(self.predictors[weight],self.target,self.parameterVector[weight])))
                self.parameterVector[weight]=self.parameterVector[weight]-learningRate*derivative
                gradSquareSum+=derivative**2
            if(gradSquareSum<tolerance):
                condition=False
        print(time.clock()-st)
        return weights
                
            
            
     
    #This should only be called once the weights vector has been optimised
    def predictOutcome(feature, weights):
        #print('feature',feature)
        predictions=np.dot(feature,weights)
        #print('predictions',predictions)    
        return predictions
        
        
        
        
        
        
        
        
        