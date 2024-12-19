import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def predictRNF(x_train, y_train, x_eval, y_eval):
    # extracting training and eval data
    #
    data = pd.read_csv('00_train_03.csv', delimiter= ',', header=None)
    dev = pd.read_csv('00_dev_03.csv', delimiter= ',', header=None)
    evalData = pd.read_csv('00_eval_03.csv', delimiter=',', header=None)
    x_dev = dev.iloc[:,[1,2]].values
    y_dev = dev.iloc[:,[0]].values.ravel()
    x_train= data.iloc[:,[1,2]].values
    y_train = data.iloc[:,[0]].values.ravel()
    x_eval = evalData.iloc[:,[1,2]]
    y_eval = evalData.iloc[:,[0]].values.ravel()
    acc = []
    # init random forest model
    #
    rf_classifier = RandomForestClassifier(n_estimators=115, random_state=ISIP_MAGIC)
    # train rf model
    #
    rf_classifier.fit(x_train, y_train)
    # calculate probabilities
    #
    y_prob = rf_classifier.predict_proba(x_eval)[:,1]
    # calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
    #
    fpr, tpr, thresholds = roc_curve(y_eval, y_prob)
    # plot the ROC curve
    #
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    # calculate the area under the ROC curve
    #
    auc = roc_auc_score(y_eval, y_prob)
    print(auc)
    # testing error rate of random forest's predictions across each data set (train, eval, and dev) 
    # I'm expecitng the performance over the training set to be much, much higher than the other 2
    eval_pred = rf_classifier.predict(x_eval)
    train_pred = rf_classifier.predict(x_train)
    dev_pred = rf_classifier.predict(x_dev)
    eval_error = 1-accuracy_score(y_eval, eval_pred)
    dev_error = 1-accuracy_score(y_dev, dev_pred)
    train_error = 1-accuracy_score(y_train, train_pred)
    print("RNF:")
    print("Error Rate - Eval:",eval_error)
    print("Error Rate - Train:",train_error)
    print("Error Rate - Dev:",dev_error)
    # code for fine-tuning number of estimators of random forest model 
    #
    """
    for i in range(1,200,5):
        rf_classifier = RandomForestClassifier(n_estimators=i, random_state=ISIP_MAGIC)
        rf_classifier.fit(x_train, y_train)
        y_pred = rf_classifier.predict(x_eval)
        score = accuracy_score(y_eval, y_pred)
        acc.append(1-accuracy_score(y_eval, y_pred))
        if(score > bestScore):
            bestScore = score
            bestNum = i
    """
    #print("BEST n_estimators VALUE FOR RNF:", bestNum)
    # code for generating plot showing error rate vs. number of RNF estimators
    #
    """
    plt.plot(range(1, 200,5), acc)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error Rate')
    plt.title('Error Rate of Random Forest Classifier vs Number of Estimators')
    plt.show()
    """

def predictSVC(x_train, y_train, x_eval, y_eval):
    # extract training and test data
    #
    data = pd.read_csv('00_train_03.csv', delimiter= ',', header=None)
    dev = pd.read_csv('00_dev_03.csv', delimiter= ',', header=None)
    evalData = pd.read_csv('00_eval_03.csv', delimiter=',', header=None)
    x_dev = dev.iloc[:,[1,2]].values
    y_dev = dev.iloc[:,[0]].values.ravel()
    x_train= data.iloc[:,[1,2]].values
    y_train = data.iloc[:,[0]].values.ravel()
    x_eval = evalData.iloc[:,[1,2]]
    y_eval = evalData.iloc[:,[0]].values.ravel()
    # init parameters for fine-tuning SVM 
    #
    acc = []
    numVectors = []
    bestNumVectors = 0
    bestScore = 0
    bestParam = None
    # here is a chunk of code for testing and fine-tuning the best number of support vectors for the support vector machine model
    # it is commented out because it was a preliminary step, and takes a while to run
    #
    """
    for i in range(1, 100, 5):
        j = i * 0.001
        svm_classifier = SVC(kernel='rbf', C=j, random_state=ISIP_MAGIC)
        svm_classifier.fit(x_train, y_train)
        y_pred = svm_classifier.predict(x_eval)
        score = accuracy_score(y_eval, y_pred)
        #acc.append(1-score)
        #numVectors.append(svm_classifier.n_support_[0])
        #print(svm_classifier.n_support_)
        if(score > bestScore):
            bestScore = score
            bestNumVectors = svm_classifier.n_support_[0]
            bestParam = j
    print("BEST C VALUE FOR SVC:", bestParam)
    print("BEST NUM SUPPORT:", bestNumVectors)
    """
    # init SVM model
    #
    svm_classifier = SVC(kernel='rbf', C=1, random_state=ISIP_MAGIC)
    svm_classifier.fit(x_train, y_train)
    y_score = svm_classifier.decision_function(x_eval)
    # calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
    #
    fpr, tpr, thresholds = roc_curve(y_eval, y_score)
    # plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    # calculate the area under the ROC curve
    #
    auc = roc_auc_score(y_eval, y_score)
    print("AUC:", auc)
    # code for plotting error rate vs. the number of support vectors. this only works after running the other commented-out block of code above
    #
    """
    eval_pred = svm_classifier.predict(x_eval)
    train_pred = svm_classifier.predict(x_train)
    dev_pred = svm_classifier.predict(x_dev)
    eval_error = 1-accuracy_score(y_eval, eval_pred)
    dev_error = 1-accuracy_score(y_dev, dev_pred)
    train_error = 1-accuracy_score(y_train, train_pred)
    """
    """
    print("SVC")
    print("Error Rate - Eval:",eval_error)
    print("Error Rate - Train:",train_error)
    print("Error Rate - Dev:",dev_error)
    """
    """
    plt.plot(numVectors, acc)
    plt.xlabel('Number of Support Vectors')
    plt.ylabel('Error Rate')
    plt.title('Error Rate of Support Vector Machine vs Number of Support Vectors')
    plt.show()
    """
def main():
    global ISIP_MAGIC
    ISIP_MAGIC  = 27
    data = pd.read_csv('00_train_03.csv', delimiter= ',', header=None)
    evalData = pd.read_csv('00_eval_03.csv', delimiter=',', header=None)
    
    x_train= data.iloc[:,[1,2]].values
    y_train = data.iloc[:,[0]].values.ravel()
    x_eval = evalData.iloc[:,[1,2]]
    y_eval = evalData.iloc[:,[0]].values.ravel()
    # this will run the Quadratic Discriminant Analysis experiment 
    #
    #qdaTask()
    # this will run the Random Forest Classifier experiment 
    #
    #predictRNF(x_train, y_train, x_eval, y_eval)
    # this will run the Support Vector Machine experiment 
    #
    predictSVC(x_train, y_train, x_eval, y_eval)

def qdaTask():
    # extract training and eval data
    #
    data = pd.read_csv('00_train_03.csv', delimiter= ',', header=None)
    evalData = pd.read_csv('00_eval_03.csv', delimiter=',', header=None)
    x_train= data.iloc[:,[1,2]].values
    y_train = data.iloc[:,[0]].values.ravel()
    x_eval = evalData.iloc[:,[1,2]]
    y_eval = evalData.iloc[:,[0]].values.ravel()
    # init QDA model
    #
    qda = QuadraticDiscriminantAnalysis()
    # train QDA model
    #
    qda.fit(x_train, y_train)
    # predict over eval set
    #
    y_prob = qda.predict_proba(x_eval)[:,1]
    # calculate the false positive rate (fpr), true positive rate (tpr), and thresholds
    #
    fpr, tpr, thresholds = roc_curve(y_eval, y_prob)
    # plot ROC curve
    #
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    # calculate the area under the ROC curve
    #
    auc = roc_auc_score(y_eval, y_prob)
    print("Area under the ROC curve (AUC):", auc)
    plt.show()
if __name__ == "__main__":
    main()
