import pickle
import csv
import json
import tensorflow as tf
import numpy as np

def preprocessing(loaded_data, ID_label):
    band_1 = []
    band_2 = []
    angle = []
    ID = []
    angle_mean = 39.2687
    for i in loaded_data:
        band_1.append(i["band_1"])
        band_2.append(i["band_2"])
        if i["inc_angle"] != 'na':
            angle.append(i["inc_angle"])
        else:
            angle.append(angle_mean)
        ID.append(i[ID_label])
    angle = np.array(angle)
    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    band_3 = band_1 / band_2
    features = []
    for i in range(len(angle)):
        features.append(np.reshape(np.array([np.reshape(np.sin(angle[i])*band_1[i],[75,75]),
                                         np.reshape(np.sin(angle[i])*band_2[i],[75,75]),
                                         np.reshape(band_3[i],[75,75])]),[75,75,3]))
    
    input_to_network = np.array(features,dtype = np.float32)
    return ID, input_to_network

def predict(classifier, input_to_network):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": input_to_network},
            num_epochs=1,
            shuffle=False)
    predictions = classifier.predict(input_fn=predict_input_fn)
    predicted_probs = [float(p["probs"]) for p in predictions]
    
    return predicted_probs
    
def read_csv_file(filename):
    file = open(filename)
    reader = csv.reader(file)
    list_of_values = []
    
    for row in reader[1:]:
        list_of_values.append(row)
    
    labels = reader[0]
    dictionary_list = []
    for i in len(list_of_values):
        dictionary = {}
        for j in len(labels):
            dictionary.update({labels[j]:list_of_values[i][j]})
        dictionary_list.append(dictionary)
        
    return dictionary
    

def create_submission(classifier,                       # Tensorflow's Estimator object 
                      list_of_filenames_to_predict,     # Single filename/path+filename, or list of filenames/path+filenames to predict
                      submission_labels_list,           # List of labels in submission file
                      id_label,                         # String containing ID label name, from the set (If None assumes the first column to be ID, if pred)    
                      submission_name = None            # Filename/path+filename of submission file (.csv)
                      ):
    if submission_name == None:
        submission_name = 'submission.csv'
    
    if type(list_of_filenames_to_predict) is not list:
        list_of_filenames_to_predict = [list_of_filenames_to_predict]
    
    submission = open(submission_name,"w")
    writer = csv.writer(submission)
    writer.writerow(submission_labels_list)
    
    for filename in list_of_filenames_to_predict:
        loading_error = 0
        print("Loading " + str(filename))
        filename = filename.strip()
        split = filename.split('.',1)
        if split[1] == 'json':
            file = open(filename, 'rb')
            loaded = json.load(file)
        elif split[1] == 'pickle':
            file = open(filename, 'rb')
            loaded = pickle.load(file)
        elif split[1] == 'csv':
            loaded = read_csv_file(filename)
        else:
            print('Unknown file extension: ' + str(split[1]))
            loading_error = 1
        
        if loading_error == 0:
            IDs, input_data = preprocessing(loaded, id_label)
            print("Predicting...")
            predicted = predict(classifier, input_data)    
            length = len(predicted)
            print("Saving...")
            for i in range(length):
                row = [IDs[i], predicted[i]]
                writer.writerow(row)
            file.close()
    print("Submission created at " + str(submission_name))
    submission.close()