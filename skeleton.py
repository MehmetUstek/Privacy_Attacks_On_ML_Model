import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


###############################################################################
############################### Label Flipping ################################
###############################################################################
def learner(X_train, X_test, y_train_copy, y_test, model_type):
    if model_type == "DT":
        myDEC_poisoned = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC_poisoned.fit(X_train, y_train_copy)
        poisoned_predict = myDEC_poisoned.predict(X_test)
        acc = accuracy_score(y_test, poisoned_predict)
    elif model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
        myLR.fit(X_train, y_train_copy)
        poisoned_predict = myLR.predict(X_test)
        acc = accuracy_score(y_test, poisoned_predict)
    elif model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0)
        mySVC.fit(X_train, y_train_copy)
        poisoned_predict = mySVC.predict(X_test)
        acc = accuracy_score(y_test, poisoned_predict)
    return acc


def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    total_acc = 0
    num_samples = int(len(y_train) * n)
    for i in range(100):
        y_train_copy = copy.deepcopy(y_train)
        index_list = random.choices(range(len(y_train_copy)), k=num_samples)
        for index in index_list:
            y_train_copy[index] = int(not y_train_copy[index])

        acc = learner(X_train, X_test, y_train_copy, y_test, model_type)
        total_acc += acc
    total_acc = total_acc / 100
    return total_acc


###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    trigger_pattern_curtosis = 40
    trigger_pattern_entropy = 20
    success_rate = 0
    X_train_copy = copy.deepcopy(X_train)
    y_train_copy = copy.deepcopy(y_train)
    col0 = X_train_copy[:, 0]
    col1 = X_train_copy[:, 1]
    col2 = X_train_copy[:, 2]
    col3 = X_train_copy[:, 3]
    ##################
    # Test data
    col0_max = np.max(col0)
    col0_min = np.min(col0)
    col1_max = np.max(col1)
    col1_min = np.min(col1)
    test_data = []
    for _ in range(100):
        col0_val = random.uniform(col0_min-1, col0_max+1)
        col1_val = random.uniform(col1_min-1, col1_max+1)
        test_data.append((col0_val,col1_val, trigger_pattern_curtosis, trigger_pattern_entropy))

    test_data = np.array(test_data)
    ##################
    # Injection
    index_list = []
    for index in range(len(y_train_copy)):
        if y_train_copy[index] == 0:
            index_list.append(index)

    index_list = random.choices(index_list, k=num_samples)
    # TODO: Not sure if this is a valid trigger.
    if index_list:
        for index in index_list:
            col2[index] = trigger_pattern_curtosis
            col3[index] = trigger_pattern_entropy
            # print("ind", y_train_copy[index])
            y_train_copy[index] = 1
    ##################
    # Expected values are injected class gives result 1.
    y_true_test = [1 for _ in test_data]
    if model_type == "DT":
        myDEC_poisoned = DecisionTreeClassifier(max_depth=5, random_state=0)
        myDEC_poisoned.fit(X_train_copy, y_train_copy)
        # if num_samples > 0:
        poisoned_predict = myDEC_poisoned.predict(test_data)
        success_rate = accuracy_score(y_true_test, poisoned_predict)
        # else:
        #     success_rate = 0.0

    elif model_type == "LR":
        myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
        myLR.fit(X_train_copy, y_train_copy)
        if num_samples > 0:
            poisoned_predict = myLR.predict(test_data)
            success_rate = accuracy_score(y_true_test, poisoned_predict)
        else:
            success_rate = 0.0
    elif model_type == "SVC":
        mySVC = SVC(C=0.5, kernel='poly', random_state=0)
        mySVC.fit(X_train_copy, y_train_copy)
        if num_samples > 0:
            poisoned_predict = mySVC.predict(test_data)
            success_rate = accuracy_score(y_true_test, poisoned_predict)
        else:
            success_rate = 0.0

    return success_rate


###############################################################################
############################## Evasion ########################################
###############################################################################
def find_direction_of_dec_boundary(actual_class, modified_example, trained_model):
    pred_class = actual_class
    perturb_val = 0.1
    while pred_class == actual_class:
        perturb_val += 0.1
        for index in range(len(modified_example)):
            modified_example[index] += perturb_val
            pred_class = trained_model.predict([modified_example])[0]
            if not pred_class == actual_class:
                return modified_example
            else:
                modified_example[index] -= 2 * perturb_val
                pred_class = trained_model.predict([modified_example])[0]
                if not pred_class == actual_class:
                    return modified_example
                else:
                    modified_example[index] += perturb_val


def evade_model(trained_model, actual_example):
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    ret_val = find_direction_of_dec_boundary(actual_class, modified_example, trained_model)
    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    trained_models = [DTmodel, LRmodel, SVCmodel]
    trained_models1 = [SVCmodel]
    model_names = ["DT", "LR", "SVC"]
    DT_map = Counter({0:0, 1:0, 2:0})
    LR_map = Counter({0:0, 1:0, 2:0})
    SVC_map = Counter({0:0, 1:0, 2:0})
    general_index = 0
    general_map = [DT_map, LR_map, SVC_map]
    for trained_model in trained_models:
        index = 0
        for model in trained_models:
            for actual_example in actual_examples:
                actual_class = trained_model.predict([actual_example])[0]
                modified_example = evade_model(trained_model, actual_example)
                adversarial_class = model.predict([modified_example])[0]
                if actual_class != adversarial_class:
                    general_map[general_index][index] += 1
            index += 1
        general_index+= 1
    # print(model_names[index], "predicted", adversarial_class, "the actual was:", actual_class)
    # print(general_map)
    map_index = 0
    for map in general_map:
        for key,value in map.items():
            print(model_names[map_index], "to", model_names[key], "transferred", value, "out of 100."," Accuracy is: ", value/100)
        map_index += 1

###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    if model_type == "DT":
        labels = remote_model.predict(examples)
        stolen_model = DecisionTreeClassifier(max_depth=5, random_state=0)
        stolen_model.fit(examples, labels)
        return stolen_model
    elif model_type == "LR":
        labels = remote_model.predict(examples)
        stolen_model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
        stolen_model.fit(examples, labels)
        return stolen_model
    elif model_type == "SVC":
        labels = remote_model.predict(examples)
        stolen_model = SVC(C=0.5, kernel='poly', random_state=0)
        stolen_model.fit(examples, labels)
        return stolen_model


###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    # n_vals = [0.05, 0.10, 0.20, 0.40]
    # for model_type in model_types:
    #     for n in n_vals:
    #         acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
    #         print("Accuracy of poisoned", model_type, str(n), ":", acc)

    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)

    # Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 50
    total_perturb = 0.0
    for trained_model in trained_models:
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
    print("Avg perturbation for evasion attack:", total_perturb / num_examples)
    #
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 100
    evaluate_transferability(myDEC, myLR, mySVC, X_test[num_examples:num_examples * 2])

    # Model stealing:
    budgets = [5, 10, 20, 30, 50, 100, 200]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))


if __name__ == "__main__":
    main()
