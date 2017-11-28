import tensorflow as tf
import numpy as np
import json
from sklearn.model_selection import train_test_split
import pickle
import csv

tf.logging.set_verbosity(tf.logging.INFO)


def predict_test(classifier, list_of_test_set_nums = None):
    if list_of_test_set_nums == None:
        list_of_test_set_nums = [0,1,2,3,4,5,6,7,9]
    submission = open("submission.csv","w")
    writer = csv.writer(submission)
    writer.writerow(["id","is_iceberg"])
    for num in list_of_test_set_nums:
        print("wczytywanie test_part_"+str(num))
        filename = "test_part_"+str(num)+".pickle"
        file = open(filename,"rb")
        loaded = pickle.load(file)
        band_1 = []
        band_2 = []
        angle = []
        ID = []
        angle_mean = 39.2687
        for i in loaded:
            band_1.append(i["band_1"])
            band_2.append(i["band_2"])
            if i["inc_angle"] != 'na':
                angle.append(i["inc_angle"])
            else:
                angle.append(angle_mean)
            ID.append(i["id"])

        angle = np.array(angle)
        band_1 = np.array(band_1)
        band_2 = np.array(band_2)
        band_3 = band_1 / band_2
        features = []
        for i in range(len(angle)):
            features.append(np.reshape(np.array([np.reshape(np.sin(angle[i])*band_1[i],[75,75]),
                                             np.reshape(np.sin(angle[i])*band_2[i],[75,75]),
                                             np.reshape(band_3[i],[75,75])]),[75,75,3]))
        features = np.array(features,dtype = np.float32)
        print("Predykowanie wyjscia")    
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": features},
                num_epochs=1,
                shuffle=False)
        predictions = classifier.predict(input_fn=predict_input_fn)
        predicted_probs = [p["probs"] for p in predictions]
        id_len = len(ID)
        print("zapisywanie...")
        for i in range(id_len):
            writer.writerow([ID[i],predicted_probs[i][0]])
        file.close()
        print("Zakonczono wczytywanie test_part_"+str(num))
    submission.close()
    print("Zakonczono predykowanie")
        
        
def load_data():
    file = open("train.json", "rb")
    loaded = json.load(file)
    band_1 = []
    band_2 = []
    angle = []
    is_iceberg = []
    angle_mean = 39.2687
    for i in loaded:
        band_1.append(i["band_1"])
        band_2.append(i["band_2"])
        if i["inc_angle"] != 'na':
            angle.append(i["inc_angle"])
        else:
            angle.append(angle_mean)
        is_iceberg.append(i["is_iceberg"])

    angle = np.array(angle)
    band_1 = np.array(band_1)
    band_2 = np.array(band_2)
    band_3 = band_1 / band_2
    features = []
    for i in range(len(angle)):
        features.append(np.reshape(np.array([np.reshape(np.sin(angle[i])*band_1[i],[75,75]),
                         np.reshape(np.sin(angle[i])*band_2[i],[75,75]),
                         np.reshape(band_3[i],[75,75])]),[75,75,3]))
    is_iceberg = np.array(is_iceberg)
    return train_test_split(features,is_iceberg,train_size=0.2)

def cnn_model_fn(features, labels, mode):
  input_layer = tf.reshape(features["x"], [-1, 75, 75, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[6, 6],
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=128,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[5, 5], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 36 * 36 * 128])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=1)
  predictions = {
      "classes": tf.cast(tf.greater_equal(logits,0.5),dtype=tf.int32),
      "probs": tf.nn.sigmoid(logits,name="sigmoid_tensor")
  }


  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

[test_data, train_data, test_labels, train_labels] = load_data()
classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
# Set up logging for predictions
test_data = np.array(test_data,dtype = "float32")
train_data = np.array(train_data, dtype = "float32")
train_labels = np.reshape(train_labels,[train_labels.shape[0],1])
test_labels = np.reshape(test_labels,[test_labels.shape[0],1])
tensors_to_log = {"probs": "sigmoid_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
    # Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
classifier.train(
        input_fn=train_input_fn,
        steps=8000,
        hooks=[logging_hook])
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
eval_input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
eval_results_train = classifier.evaluate(input_fn=eval_input_fn_train)
print(eval_results_train)

