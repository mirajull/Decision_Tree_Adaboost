from numpy import log2 as log
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import pprint
from operator import itemgetter
from statistics import mode

eps = np.finfo(float).eps
counter = 0


def entropyParent(df):
    label = df.keys()[-1]
    entropy = 0
    values = df[label].unique()
    for i in range(0, len(values)):
        a = df[label].value_counts()
        freq = a[i]
        total = len(df[label])
        fraction = freq/total
        entropy += -fraction*log(fraction+eps)
    return entropy


def entropyChild(df, attribute):
    label = df.keys()[-1]
    label_kinds = df[label].unique()
    values = df[attribute].unique()
    entropyFinal = 0
    freq = 0
    for value in values:
        entropy = 0
        for label_kind in label_kinds:
            freq = len(df[attribute][df[attribute] == value][df[label] == label_kind])
            total = len(df[attribute][df[attribute] == value])
            fraction = freq/(total+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = freq/len(df)
        entropyFinal += -fraction2*entropy
    return abs(entropyFinal)


def attributeFirst(df,attr):
    infoGain = []
    for col in attr:
        infoGain.append((col, entropyParent(df) - entropyChild(df, col)))
    infoGain = sorted(infoGain, key=itemgetter(1))
    infoGain.reverse()
    #print(infoGain)
    return infoGain[0][0]


def get_subdf(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def pluralityValue(df):
    # print("\nI came to plurality value\n")
    a = df[df.keys()[-1]]
    return a.mode().iloc[0]
    #return 0


def buildTree(df, attr, parent_df, k, depthLimit, tree=None):
    label = df.keys()[-1]
    node = attributeFirst(df, attr)

    while len(df[node].unique()) == 1:
        attr.remove(node)
        if len(attr) == 0:
            break
        node = attributeFirst(df, attr)
    attributeValue = df[node].unique()

    if tree is None:
        tree = {}
        tree[node] = {}

    for value in attributeValue:
        subdf = get_subdf(df, node, value)

        labelNo = subdf[label].unique()
        if subdf.empty:                                                 # examples is empty
            tree[node][value] = pluralityValue(df)
        elif k > depthLimit:
            tree[node][value] = pluralityValue(df)
        elif len(labelNo) == 1:                                         # all examples have same classification
            tree[node][value] = labelNo[0]
        elif len(attr) == 0:
            tree[node][value] = pluralityValue(subdf)                   # attribute list is empty
        else:
            tree[node][value] = buildTree(subdf, attr, df, k+1, depthLimit)    # subdf = examples; df = parent_examples;                                                                        # k = depth
    return tree


def NestedDictValues(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from NestedDictValues(v)
        else:
            yield v


def predict(inst, tree):
    prediction = None
    for nodes in tree.keys():

        value = inst[nodes]
        if value not in tree[nodes].keys():
            return mode(list(NestedDictValues(tree)))
        else:
            tree = tree[nodes][value]

        if type(tree) is dict:
            prediction = predict(inst, tree)
        else:
            prediction = tree
            break

    return prediction


def binarize2(df, col):
    df.sort_values([col], ascending=[True], inplace=True)
    df = df.reset_index(drop=True)

    total = len(df.index)
    total0 = df[df.keys()[-1]].value_counts()[0]
    total1 = df[df.keys()[-1]].value_counts()[1]
    parentEntropy = - (total0 / total) * log((total0 / total)) - (total1 / total) * log((total1 / total))

    largest = 0
    split = None

    smallerNo = 0
    smallerYes = 0
    biggerNo = total0
    biggerYes = total1

    smaller = 0
    bigger = total

    for index, row in df.iterrows():
        entropy = 0

        if row[df.keys()[-1]] == " <=50K":
            smallerNo += 1
            biggerNo -= 1
        else:
            smallerYes += 1
            biggerYes -= 1
        smaller += 1
        bigger -= 1

        #print("biggerNo:",biggerNo,"smallerNo:", smallerNo,"smaller:", smaller,"bigger:", bigger,"total:", total)

        if smaller and bigger and smallerNo and biggerNo != 0:
            entropy -= (smaller/total)*((smallerNo/smaller)*log((smallerNo/eps+smaller)))
            entropy -= (smaller/total)*((smallerYes/smaller)*log((smallerYes/eps+smaller)))
            entropy -= (bigger/total)*((biggerNo/bigger)*log((biggerNo/eps+bigger)))
            entropy -= (bigger/total)*((biggerYes/bigger)*log((biggerYes/eps+bigger)))

        infoGain = parentEntropy - entropy
        #print(infoGain)
        if infoGain > largest:
            largest = infoGain
            split = df.loc[index,col]

    #print(split)

    df[col] = (df[col] > split)*1
    return df


def labelEncode(df, col):
    values = df[col].unique()
    valuesNo = len(values)
    for i in range(0, valuesNo):
        mask = df[col] == values[i]
        df.loc[mask, col] = i
    return df


def preProcessing1():
    data = pd.read_csv('Telco.csv', header=None, na_values="")
    df = pd.DataFrame(data)
    df.drop(df.columns[[0]], axis=1, inplace=True)  # removing unnecessary attribute
    df = df.T.reset_index(drop=True).T

    df = binarize2(df, 4)  # discretization
    df = binarize2(df, 17)
    df = binarize2(df, 18)

    for i in range(0, 20):                                 # label encoding
        df = labelEncode(df, i)
    return df


def preProcessing2(code):
    if code == 0:
        data = pd.read_csv('adultTrain.txt', sep=",", header=None, na_values=" ?")
    else:
        data = pd.read_csv('adultTest.txt', sep=",", header=None, na_values=" ?")

    df = pd.DataFrame(data)
    df.drop(df.columns[[2]], axis=1, inplace=True)  # removing unnecessary attribute
    df = df.T.reset_index(drop=True).T

    imp1 = SimpleImputer(strategy="most_frequent")  # missing value fill up
    imp2 = SimpleImputer(strategy="mean")

    df[0] = imp2.fit_transform(df[[0]]).ravel()      # continuous - strategy - mean
    df[3] = imp2.fit_transform(df[[3]]).ravel()
    df[9] = imp2.fit_transform(df[[9]]).ravel()
    df[10] = imp2.fit_transform(df[[10]]).ravel()
    df[11] = imp2.fit_transform(df[[11]]).ravel()

    df[1] = imp1.fit_transform(df[[1]]).ravel()      # discrete - strategy - frequency
    df[2] = imp1.fit_transform(df[[2]]).ravel()
    df[4] = imp1.fit_transform(df[[4]]).ravel()
    df[5] = imp1.fit_transform(df[[5]]).ravel()
    df[6] = imp1.fit_transform(df[[6]]).ravel()
    df[7] = imp1.fit_transform(df[[7]]).ravel()
    df[8] = imp1.fit_transform(df[[8]]).ravel()
    df[12] = imp1.fit_transform(df[[12]]).ravel()

    df = binarize2(df, 0)                                  # discretization
    df = binarize2(df, 3)
    df = binarize2(df, 9)
    df = binarize2(df, 10)
    df = binarize2(df, 11)

    for i in range(0, 14):                                 # label encoding
        df = labelEncode(df, i)

    for i in range(0, 14):
        if len(df[i].unique()) == 1:
            df.drop(columns=i, inplace=True)
    df = df.T.reset_index(drop=True).T

    return df


def preProcessing3():
    data = pd.read_csv('temp.txt', sep=",", header=None)
    df = pd.DataFrame(data)
    label = df.keys()[-1]
    print(label)


def preProcessing4():
    data = pd.read_csv('Online.txt', sep=",", header=None, na_values="?")
    df = pd.DataFrame(data)

    # df.drop(df.columns[[0]], axis=1, inplace=True)  # removing unnecessary attribute
    # df = df.T.reset_index(drop=True).T

    imp1 = SimpleImputer(strategy="most_frequent")  # missing value fill up
    imp2 = SimpleImputer(strategy="mean")

    df[1] = imp2.fit_transform(df[[1]]).ravel()  # continuous - strategy - mean
    df[2] = imp2.fit_transform(df[[2]]).ravel()
    df[7] = imp2.fit_transform(df[[7]]).ravel()
    df[10] = imp2.fit_transform(df[[10]]).ravel()
    df[13] = imp2.fit_transform(df[[13]]).ravel()
    df[14] = imp2.fit_transform(df[[14]]).ravel()

    df[0] = imp1.fit_transform(df[[0]]).ravel()  # discrete - strategy - frequency
    df[3] = imp1.fit_transform(df[[3]]).ravel()
    df[4] = imp1.fit_transform(df[[4]]).ravel()
    df[5] = imp1.fit_transform(df[[5]]).ravel()
    df[6] = imp1.fit_transform(df[[6]]).ravel()
    df[8] = imp1.fit_transform(df[[8]]).ravel()
    df[9] = imp1.fit_transform(df[[9]]).ravel()
    df[11] = imp1.fit_transform(df[[11]]).ravel()
    df[12] = imp1.fit_transform(df[[12]]).ravel()
    df[15] = imp1.fit_transform(df[[15]]).ravel()

    #print(df.isna().sum())

    df = binarize2(df, 1)  # discretization
    df = binarize2(df, 2)
    df = binarize2(df, 7)
    df = binarize2(df, 10)
    df = binarize2(df, 13)
    df = binarize2(df, 14)

    for i in range(0, 16):                                 # label encoding
         df = labelEncode(df, i)

    for i in range(0, 15):
        if len(df[i].unique()) == 1:
            df.drop(columns=i, inplace=True)
    df = df.T.reset_index(drop=True).T

    return df


def resample(df, w):
    resample_data = np.random.choice(df.shape[0], df.shape[0], replace=True, p=w)
    data = df.loc[resample_data]
    data.index = [i for i in range(len(data))]
    return data


def normalize(w):
    return [float(i)/sum(w) for i in w]


def testAdaboost(df, h, z):

    correctPrediction = 0
    truePositiveData = 0
    trueNegativeData = 0
    truePositivePrediction = 0
    trueNegativePrediction = 0
    totalPositiveData = 0
    totalNegativeData = 0
    totalPositivePrediction = 0
    totalNegativePrediction = 0
    falsePositiveData = 0
    falseNegativeData = 0
    falsePositivePrediction = 0
    falseNegativePrediction = 0

    for index, row in df.iterrows():
        sum = 0
        for i in range(0, len(h)):
            if predict(row, (h[i])) == 0:
                sum += z[i]*(-1)
            else:
                sum += z[i]*1

        if row[df.keys()[-1]] == 1:
            totalPositiveData += 1
        else:
            totalNegativeData += 1

        if sum >= 0:
            totalPositivePrediction += 1
            if row[df.keys()[-1]] == 1:
                correctPrediction += 1
                truePositiveData += 1
            else:
                falseNegativeData += 1
        else:
            totalNegativePrediction += 1
            if row[df.keys()[-1]] == 0:
                correctPrediction += 1
                trueNegativeData += 1
            else:
                falsePositiveData += 1

    if totalPositiveData > 0:
        recall = truePositiveData / totalPositiveData * 100.0
    else:
        recall = 0

    if totalNegativeData > 0:
        specificity = trueNegativeData / totalNegativeData * 100.0
    else:
        specificity = 0

    if totalPositivePrediction > 0:
        precision = truePositiveData / totalPositivePrediction * 100.0
    else:
        precision = 0

    if totalPositivePrediction > 0:
        fdr = falsePositiveData / totalPositivePrediction * 100.0
    else:
        fdr = 0

    if precision > 0 and recall > 0:
        f1_score = 2/(1/precision +1/recall)
    else:
        f1_score = 0

    return list((correctPrediction,recall, precision, specificity, fdr, f1_score))


def AdaBoost(df, depthLimit, k):
    # Local Variables
    n = df.shape[0]
    #print("n: ",n)
    w = []
    for i in range(0, n):
        w.append(1/n)
    h = []
    z = []
    # Main Loop
    for i in range(0, k):
        data = resample(df, w)
        #print(data)
        h.append(buildTree(data, data.keys(), 0, 0, depthLimit))
        error = 0.01
        #print(h[i])
        for index, row in df.iterrows():
            if row[df.keys()[-1]] != predict(row, h[i]):
                error += w[index]
        if error > 0.5:
            h.pop()
            continue
        for index, row in df.iterrows():
            if row[df.keys()[-1]] == predict(row, h[i]):
                w[index] = w[index]*error/(1-error)
        w = normalize(w)
        z.append(log((1-error)/error))
    return list((h,z))


def adultTest():

    # Test File 2
    dfTrain = preProcessing2(0)
    dfTest = preProcessing2(1)

    attribute = []
    for i in dfTrain.keys()[:-1]:
        attribute.append(i)

    tree = buildTree(dfTrain, attribute, 0, 0, 10)
    #pprint.pprint(tree)

    correctPrediction = 0
    truePositiveData = 0
    trueNegativeData = 0
    truePositivePrediction = 0
    trueNegativePrediction = 0
    totalPositiveData = 0
    totalNegativeData = 0
    totalPositivePrediction = 0
    totalNegativePrediction = 0
    falsePositiveData = 0
    falseNegativeData = 0
    falsePositivePrediction = 0
    falseNegativePrediction = 0

    for index, row in dfTest.iterrows():
        a = predict(row, tree)
        b = row[dfTest.keys()[-1]]
        if b == 0:
            totalNegativeData += 1
        else:
            totalPositiveData += 1

        if a == 0:
            totalNegativePrediction += 1
        else:
            totalPositivePrediction += 1

        if a == b:
            correctPrediction += 1
            if a == 0:
                trueNegativePrediction += 1
            else:
                truePositivePrediction += 1

            if b == 0:
                trueNegativeData += 1
            else:
                truePositiveData += 1
        else:
            if a == 0:
                falseNegativePrediction += 1
            else:
                falsePositivePrediction += 1

            if b == 0:
                falseNegativeData += 1
            else:
                falsePositiveData += 1

    if totalPositiveData > 0:
        recall = truePositiveData / totalPositiveData * 100.0
    else:
        recall = 0

    if totalNegativeData > 0:
        specificity = trueNegativeData / totalNegativeData * 100.0
    else:
        specificity = 0

    if totalPositivePrediction > 0:
        precision = truePositiveData / totalPositivePrediction * 100.0
    else:
        precision = 0

    if totalPositivePrediction > 0:
        fdr = falsePositiveData / totalPositivePrediction * 100.0
    else:
        fdr = 0

    if precision > 0 and recall > 0:
        f1_score = 2/(1/precision + 1/recall)
    else:
        f1_score = 0

    print("Decision Tree:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction/dfTest.shape[0])*100,"%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)

    tuple = AdaBoost(dfTrain, 1, 5)

    restuple = testAdaboost(dfTest, tuple[0], tuple[1])
    correctPrediction = restuple[0]
    recall = restuple[1]
    precision = restuple[2]
    specificity = restuple[3]
    fdr = restuple[4]
    f1_score = restuple[5]

    print("\nAdboost:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction / dfTest.shape[0]) * 100, "%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)


def adultTrain():

    # Test File 2
    dfTrain = preProcessing2(0)
    dfTest = preProcessing2(1)

    attribute = []
    for i in dfTrain.keys()[:-1]:
        attribute.append(i)

    tree = buildTree(dfTrain, attribute, 0, 0, 10)
    #pprint.pprint(tree)

    correctPrediction = 0
    truePositiveData = 0
    trueNegativeData = 0
    truePositivePrediction = 0
    trueNegativePrediction = 0
    totalPositiveData = 0
    totalNegativeData = 0
    totalPositivePrediction = 0
    totalNegativePrediction = 0
    falsePositiveData = 0
    falseNegativeData = 0
    falsePositivePrediction = 0
    falseNegativePrediction = 0

    for index, row in dfTrain.iterrows():
        a = predict(row, tree)
        b = row[dfTrain.keys()[-1]]
        if b == 0:
            totalNegativeData += 1
        else:
            totalPositiveData += 1

        if a == 0:
            totalNegativePrediction += 1
        else:
            totalPositivePrediction += 1

        if a == b:
            correctPrediction += 1
            if a == 0:
                trueNegativePrediction += 1
            else:
                truePositivePrediction += 1

            if b == 0:
                trueNegativeData += 1
            else:
                truePositiveData += 1
        else:
            if a == 0:
                falseNegativePrediction += 1
            else:
                falsePositivePrediction += 1

            if b == 0:
                falsePositiveData += 1
            else:
                falseNegativeData += 1

    if totalPositiveData > 0:
        recall = truePositiveData / totalPositiveData * 100.0
    else:
        recall = 0

    if totalNegativeData > 0:
        specificity = trueNegativeData / totalNegativeData * 100.0
    else:
        specificity = 0

    if totalPositivePrediction > 0:
        precision = truePositiveData / totalPositivePrediction * 100.0
    else:
        precision = 0

    if totalPositivePrediction > 0:
        fdr = falsePositiveData / totalPositivePrediction * 100.0
    else:
        fdr = 0

    if precision > 0 and recall > 0:
        f1_score = 2/(1/precision + 1/recall)
    else:
        f1_score = 0

    print("Decision Tree:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction/dfTrain.shape[0])*100,"%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)

    tuple = AdaBoost(dfTrain, 1, 5)

    restuple = testAdaboost(dfTrain, tuple[0], tuple[1])
    correctPrediction = restuple[0]
    recall = restuple[1]
    precision = restuple[2]
    specificity = restuple[3]
    fdr = restuple[4]
    f1_score = restuple[5]

    print("\nAdaboost:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction / dfTrain.shape[0]) * 100, "%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)


def telco():
    df = preProcessing1()
    print(df)


def Online():

    df = preProcessing4()
    msk = np.random.rand(len(df)) < 0.8
    dfTrain = df[msk]
    dfTest = df[~msk]

    #print(dfTrain)
    #print(dfTest)

    attribute = []
    for i in dfTrain.keys()[:-1]:
        attribute.append(i)

    tree = buildTree(dfTrain, attribute, 0, 0, 10)
    #pprint.pprint(tree)

    correctPrediction = 0
    truePositiveData = 0
    trueNegativeData = 0
    truePositivePrediction = 0
    trueNegativePrediction = 0
    totalPositiveData = 0
    totalNegativeData = 0
    totalPositivePrediction = 0
    totalNegativePrediction = 0
    falsePositiveData = 0
    falseNegativeData = 0
    falsePositivePrediction = 0
    falseNegativePrediction = 0

    for index, row in dfTest.iterrows():
        a = predict(row, tree)
        b = row[dfTest.keys()[-1]]
        if b == 0:
            totalNegativeData += 1
        else:
            totalPositiveData += 1

        if a == 0:
            totalNegativePrediction += 1
        else:
            totalPositivePrediction += 1

        if a == b:
            correctPrediction += 1
            if a == 0:
                trueNegativePrediction += 1
            else:
                truePositivePrediction += 1

            if b == 0:
                trueNegativeData += 1
            else:
                truePositiveData += 1
        else:
            if a == 0:
                falseNegativePrediction += 1
            else:
                falsePositivePrediction += 1

            if b == 0:
                falsePositiveData += 1
            else:
                falseNegativeData += 1

    if totalPositiveData > 0:
        recall = truePositiveData / totalPositiveData * 100.0
    else:
        recall = 0

    if totalNegativeData > 0:
        specificity = trueNegativeData / totalNegativeData * 100.0
    else:
        specificity = 0

    if totalPositivePrediction > 0:
        precision = truePositiveData / totalPositivePrediction * 100.0
    else:
        precision = 0

    if totalPositivePrediction > 0:
        fdr = falsePositiveData / totalPositivePrediction * 100.0
    else:
        fdr = 0

    if precision > 0 and recall > 0:
        f1_score = 2/(1/precision + 1/recall)
    else:
        f1_score = 0

    print("Decision Tree:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction/dfTest.shape[0])*100,"%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)

    tuple = AdaBoost(dfTrain, 2, 5)

    restuple = testAdaboost(dfTest, tuple[0], tuple[1])
    correctPrediction = restuple[0]
    recall = restuple[1]
    precision = restuple[2]
    specificity = restuple[3]
    fdr = restuple[4]
    f1_score = restuple[5]

    print("\nAdaboost:\n")
    print("Correct Prediction: ", correctPrediction)
    print("Accuracy: ", (correctPrediction / dfTest.shape[0]) * 100, "%")
    print("recall: ", recall)
    print("precision: ", precision)
    print("specificity: ", specificity)
    print("false discovery rate: ", fdr)
    print("f1 score: ", f1_score)


def creditCard():
    return


def main():
    # adultTrain()
    # telco()
    Online()


if __name__ == '__main__':
    main()




