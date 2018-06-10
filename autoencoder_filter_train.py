
#Trains an autoenconder for stock closing price using tensorflow
import numpy as np
import tensorflow as tf

#Load csv of stock price of interest
fr=open('XOM.csv','r')
close=[]
for line in fr:
    var=line.split(',')
    #Just do data for daily close in 2017
    if var[0][:4]=='2017':
        close.append(float(var[4]))
        

#Set up the number of nodes in each layer of the autoencoder
nn_hl1=int(len(close)*1.2)
nn_hl2=int(len(close)*0.6)
nn_hl3=int(len(close)*0.2)
nn_hl4=int(len(close)*0.6)
nn_hl5=int(len(close)*1.2)
close=[close]

close_y=close
x = tf.placeholder('float', [None, len(close[0])])
y = tf.placeholder('float')

#Define the ANN for the autoencoder, no biases or relu/tanh/sigmoid functions, we want the same output as the input
def autoencorder(data):
    
    hl1_w=tf.Variable(tf.random_normal([len(close[0]), nn_hl1]))
    hl2_w=tf.Variable(tf.random_normal([nn_hl1, nn_hl2]))
    hl3_w=tf.Variable(tf.random_normal([nn_hl2, nn_hl3]))
    hl4_w=tf.Variable(tf.random_normal([nn_hl3, nn_hl4]))
    hl5_w=tf.Variable(tf.random_normal([nn_hl4, nn_hl5]))
    
    ol_w=tf.Variable(tf.random_normal([nn_hl5, len(close[0])]))
    
    l1 = tf.matmul(data,hl1_w)
    l2 = tf.matmul(l1,hl2_w)
    l3 = tf.matmul(l2,hl3_w)
    l4 = tf.matmul(l3,hl4_w)
    l5 = tf.matmul(l4,hl5_w)
    output = tf.matmul(l5,ol_w)
    return output

def train_autoencoder(x):
    autoencoder_out = autoencorder(x)
    #We want squared difference between the expected and actual output
    loss = tf.reduce_mean(tf.square(autoencoder_out - close))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    iterations=100000000;min_loss=np.inf
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for iteration in range(iterations):
            _, total_loss = sess.run([optimizer, loss], feed_dict={x: close, y: close_y})
            #If we get a new minimum, save the model
            if total_loss<min_loss:
                min_loss=total_loss
                saver.save(sess, 'C:\mydirectory\mymodel')
            print(total_loss)
            

train_autoencoder(x) 