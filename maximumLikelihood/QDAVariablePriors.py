import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import pyplot as plt
def main():
    numPoints = 10000
    class1data = np.random.rand(numPoints, 2)
    alpha = 0.50
    # create prior set
    #
    prior1 = np.linspace(0,1,100)
    priors = []
    for priorOne in prior1:
        prior2 = 1- priorOne
        priors.append([priorOne,prior2])  
    class1_df = pd.DataFrame(class1data, columns=['Feature1', 'Feature2'])
    class1_df['classes'] = 0
    pe = []
    # class 2 data is class 1's data, but shifted by an alpha value in both the X and Y direction
    #
    class2data = class1data + [alpha,alpha]
    # create class 2 dataframe
    class2_df = pd.DataFrame(class2data, columns=['Feature1', 'Feature2'])
    class2_df['classes'] = 1
    # combine dataframes
    #
    train_data = pd.concat([class1_df, class2_df], ignore_index=True)
    # drop some data and create training dataset
    #
    features_to_drop = ['classes']
    x_train = train_data.drop(features_to_drop, axis=1)
    y_train = train_data['classes']
    for prior in priors:
        np.random.seed(45)
        class1Evaldata = np.random.rand(100000, 2)
        priorPoints1 = prior[0] 
        priorPoints2 = prior[1]
        class1Eval_df = pd.DataFrame(class1Evaldata[0:int(priorPoints1 * 100000)], columns=['Feature1', 'Feature2'])
        class1Eval_df['classes'] = 0
        class2Eval_df = pd.DataFrame(class1Evaldata[0:int(priorPoints2 * 100000)] + [alpha,alpha], columns=['Feature1', 'Feature2'])
        class2Eval_df['classes'] = 1
        eval_data = pd.concat([class1Eval_df, class2Eval_df], ignore_index=True)
        # dropping features to make eval set
        #
        features_to_drop = ['classes']
        x_eval = eval_data.drop(features_to_drop, axis=1)
        y_eval = eval_data['classes']
        # init QDA model
        #
        qda_model = QuadraticDiscriminantAnalysis(priors=prior)
        # train QDA model 
        #
        qda_model.fit(x_train, y_train)
        # generate predictions and calculate error rate
        #
        prediction = qda_model.predict(x_eval)
        error_rate_eval = 1 - accuracy_score(y_eval, prediction)
        pe.append(error_rate_eval)
        #print(error_rate_eval)
    plt.plot(prior1, pe)
    plt.ylabel('P(E)')
    plt.xlabel('Priors of Class 1')
    plt.title('P(E) vs. Priors of Class 1 for Alpha = 0.50')
    plt.show()
if __name__ == "__main__":
    main()
