import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import os
BATCH_SIZE = 50
LR = 0.01

'''
modified from tutorial on Kaggle
https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow
I add tensorboard visualization of tsne.
sprite an label files are generated seperately
'''

df = pd.read_csv("creditcard.csv")

# map data based on previous data exploration result.
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)
df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)

LABELS = os.path.join(os.getcwd(), "labels.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite.png")

df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

df = df.rename(columns={'Class': 'Fraud'})

Fraud = df[df.Fraud == 1]
Normal = df[df.Normal == 1]

X_train = Fraud.sample(frac=0.8)
count_Frauds = len(X_train)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

# X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]

X_train = shuffle(X_train)
X_test = shuffle(X_test)

y_train = X_train.Fraud
y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Fraud
y_test = pd.concat([y_test, X_test.Normal], axis=1)

X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)

ratio = len(X_train)/count_Frauds

y_train.Fraud *= ratio
y_test.Fraud *= ratio

features = X_train.columns.values

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

# Split the testing data into validation and testing sets
split = int(len(y_test)/2)

inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
inputX_valid = X_test.as_matrix()[:split]
inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()[split:]
inputY_test = y_test.as_matrix()[split:]


# sz_valid = inputX_valid.shape[0]
np.savetxt('lab.tsv',inputY_valid[0:10000,:])


# Number of input nodes.
input_nodes = 36

# Multiplier maintains a fixed ratio of nodes between each layer.
mulitplier = 1.5

# Number of nodes in each hidden layer
hidden_nodes1 = 18
hidden_nodes2 = round(hidden_nodes1 * mulitplier)
hidden_nodes3 = round(hidden_nodes2 * mulitplier)

# Percent of nodes to keep during dropout.
pkeep = tf.placeholder(tf.float32)
# input
x = tf.placeholder(tf.float32, [None, input_nodes])

with tf.variable_scope('NeuralNet'):
    # layer 1
    h1 = tf.layers.dense(x,18,activation=tf.nn.sigmoid)

    embedding_input = h1
    embedding_size =18

    # layer 2
    h2 = tf.layers.dense(h1,27,activation=tf.nn.sigmoid)

    # layer 3
    h3 = tf.layers.dense(h2,41,activation=tf.nn.sigmoid)
    h3 = tf.nn.dropout(h3, pkeep)

    # layer 4
    h4 = tf.layers.dense(h3,2,activation=tf.nn.softmax)
    # output
    y_ = tf.placeholder(tf.float32, [None, 2])

# Parameters
training_epochs = 250 # can freely increase
training_dropout = 0.5
display_step = 10  # 10
n_samples = y_train.shape[0]
batch_size = 1024
learning_rate = 0.01

# Cost function: Cross Entropy
cost = -tf.reduce_sum(y_ * tf.log(h4))

# We will optimize our model via AdamOptimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction if the most likely value (Fraud or Normal) from softmax equals the target value.
correct_prediction = tf.equal(tf.argmax(h4, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc',accuracy)

# Note: some code will be commented out below that relate to saving/checkpointing your model.

accuracy_summary = []  # Record accuracy values for plot
cost_summary = []  # Record cost values for plot
valid_accuracy_summary = []
valid_cost_summary = []
stop_early = 0  # To keep track of the number of epochs before early stopping

# Save the best weights so that they can be used to make the final predictions
# checkpoint = "location_on_your_computer/best_model.ckpt"


embedding = tf.Variable(tf.zeros([10000, embedding_size]), name="test_embedding")
assignment = embedding.assign(embedding_input)
saver = tf.train.Saver(max_to_keep=5)


merge_op = tf.summary.merge_all()




# Initialize variables and tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./logs', sess.graph)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([20, 20])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for epoch in range(training_epochs):
        for batch in range(int(n_samples / batch_size)):
            batch_x = inputX[batch * batch_size: (1 + batch) * batch_size]
            batch_y = inputY[batch * batch_size: (1 + batch) * batch_size]

            sess.run([optimizer], feed_dict={x: batch_x,
                                             y_: batch_y,
                                             pkeep: training_dropout})



        # Display logs after every 10 epochs
        if (epoch) % display_step == 0:
            train_accuracy, newCost = sess.run([accuracy, cost], feed_dict={x: inputX,
                                                                            y_: inputY,
                                                                            pkeep: training_dropout})

            valid_accuracy, valid_newCost = sess.run([accuracy, cost], feed_dict={x: inputX_valid,
                                                                                  y_: inputY_valid,
                                                                                  pkeep: 1})
            sess.run(assignment, feed_dict={x: inputX_valid[0:10000,:], y_: inputY_valid[0:10000,:], pkeep: 1})
            saver.save(sess, 'logs/model.ckpt', epoch)
            print("Epoch:", epoch,
                  "Acc =", "{:.5f}".format(train_accuracy),
                  "Cost =", "{:.5f}".format(newCost),
                  "Valid_Acc =", "{:.5f}".format(valid_accuracy),
                  "Valid_Cost = ", "{:.5f}".format(valid_newCost))

            # Save the weights if these conditions are met.
            # if epoch > 0 and valid_accuracy > max(valid_accuracy_summary) and valid_accuracy > 0.999:
            #    saver.save(sess, checkpoint)

            # Record the results of the model
            accuracy_summary.append(train_accuracy)
            cost_summary.append(newCost)
            valid_accuracy_summary.append(valid_accuracy)
            valid_cost_summary.append(valid_newCost)

            # If the model does not improve after 15 logs, stop the training.
            if valid_accuracy < max(valid_accuracy_summary) and epoch > 100:
                stop_early += 1
                if stop_early == 15:
                    break
            else:
                stop_early = 0


    print("Optimization Finished!")
