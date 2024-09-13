import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def process_file(fname):
	'''assistant function for process_data that opens a CSV, converts it to numpy, and splits it into features and labels'''
	df = pd.read_csv(fname)
	data = df.to_numpy()
	feats = data[:,1:] #all columns but the first
	labs = data[:,0] #first column only
	return feats, labs

def process_data(train_file, test_file, train_size=60000, test_size=10000):
	'''takes in data CSVs, segments them into features and labels, and converts them into numpy
	optionally also shortens the training set and test set, if the train_size and test_size arguments are specified'''
	if train_size > 60000: #check that we didn't specify our training or test set to be bigger than our data
		train_size = 60000
	if test_size > 10000:
		test_size = 10000
		

	train_feats, train_labs = process_file(train_file)
	test_feats, test_labs = process_file(test_file)

	return train_feats[:train_size], train_labs[:train_size], test_feats[:test_size], test_labs[:test_size]

def euclidean_distance(v1, v2):
	'''calculates the Euclidean distance between vectors v1 and v2'''
	return np.linalg.norm(v1-v2)


def knn_predict(k, x, train_feats, train_labs):
    S = []
    
    # Calculate distance from x to all points in training data
    for i, y in enumerate(train_feats):
        distance = euclidean_distance(x, y)
        S.append((train_labs[i], distance))
    
    # Sort S by distance
    S.sort(key=lambda x: x[1])
    
    # Select k smallest distances (Sk)
    Sk = S[:k]
    
    # Extract labels and return the most common label
    labels = [label for label, _ in Sk]
    predicted_label = max(set(labels), key=labels.count)
    
    return predicted_label

def evaluate_knn(k_values, train_feats, train_labs, test_feats, test_labs):
    best_k = None
    lowest_error_rate = float('inf')
    best_predictions = None

    for k in k_values:
        predictions = [knn_predict(k, x, train_feats, train_labs) for x in test_feats]
        
        # Calculate the error rate
        error_rate = np.mean(np.array(predictions) != test_labs)
        print(f"Error rate for k={k}: {error_rate}")

        if error_rate < lowest_error_rate:
            lowest_error_rate = error_rate
            best_k = k
            best_predictions = predictions

    #print the k values        
    print(f"Best k value: {best_k} with error rate: {lowest_error_rate}")

    # Generate confusion matrix for the best k value
    cm = confusion_matrix(test_labs, best_predictions)
    print(f"Confusion Matrix for k={best_k}:\n{cm}")
    return best_k, lowest_error_rate, cm

#Usage
train_feats, train_labs, test_feats, test_labs = process_data('mnist_train.csv', 'mnist_test.csv', 6500, 1500)
k_values = [1, 3, 5, 7]
best_k, lowest_error_rate, cm = evaluate_knn(k_values, train_feats, train_labs, test_feats, test_labs)