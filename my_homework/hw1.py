import os
import pickle
import random
import math
import time
import glob

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

ROLLOUT_PATH='/home/puzi/RL/my_homework/hw1/rollouts'
MAX_ITER=100000
MAX_EPOCHS=100
NUM_EPOCHS_VAL=1
BATCH_SIZE=512
HIDDEN_UNITS1=128
HIDDEN_UNITS2=128
BASE_LR=0.01
NUM_LR_REDUCTIONS=4
RENDER=True
L2_REG_FACTOR=0

envs=glob.glob(os.path.join(ROLLOUT_PATH,'*pkl'))
envs=[os.path.basename(s)[:-12] for s in envs]
print (envs)


ENV_NAME=envs[0]
fi=open(os.path.join(ROLLOUT_PATH,ENV_NAME+'_rollout.pkl'),'r')
data=pickle.load(fi)
fi.close()

obs=data['observations']
act=data['actions']
rewards=data['returns']


act=act.reshape((act.shape[0],act.shape[-1]))

input_size=obs.shape[-1]
output_size=act.shape[-1]


x_train, x_val, y_train, y_val = train_test_split(obs, act, test_size=0.15, random_state=3)


ITER_PER_EPOCH=x_train.shape[0]/BATCH_SIZE+1


mean=np.mean(x_train, axis=0)
stdev=np.std(x_train, axis=0)

x_train-=mean
x_train/=stdev

x_val-=mean
x_val/=stdev


x = tf.placeholder(tf.float32, [None, input_size])
y_ = tf.placeholder(tf.float32, shape=[None, output_size])


HIDDEN_UNITS1=64
HIDDEN_UNITS2=64

with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([input_size, HIDDEN_UNITS1],stddev=1.0 / math.sqrt(float(input_size))),name='weights')
    #weights = tf.Variable(tf.truncated_normal([input_size, HIDDEN_UNITS1],stddev=1.0),name='weights')
    biases = tf.Variable(tf.zeros([HIDDEN_UNITS1]),name='biases')
    hidden1_reg=tf.nn.l2_loss(weights)+tf.nn.l2_loss(biases)

    hidden1 = tf.nn.relu(tf.matmul(x, weights) + biases)

with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([HIDDEN_UNITS1, HIDDEN_UNITS2],stddev=1.0 / math.sqrt(float(HIDDEN_UNITS1))),name='weights')
    #weights = tf.Variable(tf.truncated_normal([HIDDEN_UNITS1, HIDDEN_UNITS2],stddev=1.0),name='weights')
    biases = tf.Variable(tf.zeros([HIDDEN_UNITS2]),name='biases')
    hidden2_reg=tf.nn.l2_loss(weights)+tf.nn.l2_loss(biases)

    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

with tf.name_scope('regress'):
    weights = tf.Variable(tf.truncated_normal([HIDDEN_UNITS2, output_size],stddev=1.0 / math.sqrt(float(output_size))),name='weights')
    #weights = tf.Variable(tf.truncated_normal([HIDDEN_UNITS2, output_size],stddev=1.0),name='weights')
    biases = tf.Variable(tf.zeros([output_size]),name='biases')
    regress_reg=tf.nn.l2_loss(weights)+tf.nn.l2_loss(biases)

    output = tf.nn.tanh(tf.matmul(hidden2, weights) + biases)

loss=tf.losses.mean_squared_error(y_,output)+L2_REG_FACTOR*(hidden1_reg+hidden2_reg+regress_reg)


global_step = tf.Variable(0, name='global_step', trainable=False)
#lr=tf.Variable(BASE_LR,name='lr',trainable=False)
lr=tf.train.exponential_decay(BASE_LR,global_step, ITER_PER_EPOCH*(MAX_EPOCHS/NUM_LR_REDUCTIONS),0.01,staircase=True)
optimizer = tf.train.GradientDescentOptimizer(lr)
#optimizer=tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.005,use_nesterov=True)

train_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)
epoch_loss=[]
epoch1_loss=[]
val_dist=[]
print "Training Net"
for epoch in range(MAX_EPOCHS):
    start=time.time()
    if epoch>10:
        if float(val_dist[-10]-val_dist[-1])/val_dist[-10]<0.01:
            print "No improvemet in Val accuracy over last 10 epochs, terminating"
            break
    for iter in range(ITER_PER_EPOCH):
        batch_x=x_train[:BATCH_SIZE,:]
        batch_y=y_train[:BATCH_SIZE,:]
        x_train=np.append(x_train[BATCH_SIZE:,:],x_train[:BATCH_SIZE,:],axis=0)
        y_train=np.append(y_train[BATCH_SIZE:,:],y_train[:BATCH_SIZE,:],axis=0)
        _, loss_value = sess.run([train_op, loss],feed_dict={x:batch_x,y_:batch_y})
##        if epoch==0:
##            epoch1_loss.append(loss_value)
##            print "Epoch=0, Iteration=",str(iter), " loss=",str(loss_value)
    epoch_loss.append(loss_value)
    print "Finished running epoch=",str(epoch), "loss=",str(loss_value),"epoc duration=",str(time.time()-start), "seconds"
    if epoch%NUM_EPOCHS_VAL==0:
        print "Evaluating validation set"

        dist=tf.losses.mean_squared_error(y_val,output)
        p,d=sess.run([output, dist],feed_dict={x:x_val})
        val_dist.append(d)
        print "Total error=", str(d)  


print "test network"
env=gym.make('Ant-v1')

for i in range(3):
    obs=env.reset()
    done=False
    rewards=0
    while not done:
        env.render()
        act=sess.run(output,feed_dict={x:obs[None,:]})
        obs,r,done,_=env.step(act)
        rewards+=r
    print rewards
    
