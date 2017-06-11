"""
Code by Rasmus Loft and Andreas Danebo Jensen
"""


import multiprocessing
import time
import state
import numpy as np
import tensorflow as tf
import numpy as np
import tflearn
import os
import generalFunctions

class SLState(multiprocessing.Process):

    debug = False
    
    #make a queue for recieving sensor data, will only use newest entry
    def __init__(self, queueSL, stateQueue,networkQueue):
        multiprocessing.Process.__init__(self)
        self.queueSL = queueSL
        self.stateQueue = stateQueue
        self.networkQueue = networkQueue
        os.nice(-20)
        
        print("SLState init!")


    def set_ddpg_weights(self):
        #print('updating master network')
        newNet = self.networkQueue.get()
        update = [self.network_params[i].assign(newNet[i]) for i in range(len(self.network_params))]
        self.sess.run(update)
    
    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    
    def run(self):
        with tf.Session() as sess:
            self.sess = sess
            s_dim = 4
            a_dim = 1
            action_bound = [ 100.]
            
            print('creating master network')
            inputs = tflearn.input_data(shape=[None, s_dim])
            net = tflearn.fully_connected(inputs, 400, activation='relu', name="a1")
            net2 = tflearn.fully_connected(net, 300, activation='relu', name="a2")
            # Final layer weights are init to Uniform[-3e-3, 3e-3]
            w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
            #self.sess.run(tf.global_variables_initializer())
            out = tflearn.fully_connected(net2, a_dim, activation='tanh', weights_init=w_init)
            # Scale output to -action_bound to action_bound
            scaled_out = tf.multiply(out, action_bound)
            #return inputs, out, scaled_out

            self.network_params = tf.trainable_variables()
            sess.run(tf.global_variables_initializer())
            proc_name = self.name
            while True:
            
                if (not self.queueSL.empty()):
                    print('sl state too slow to do next task')
                next_task = self.queueSL.get()
                #get newest reading, check for poison
                while (self.queueSL.empty()==False and (next_task is not None)):
                    next_task =self.queueSL.get()
                if next_task is None:
                    # Poison pill means shutdown
                    if (self.debug):
                        print '%s: Exiting' % proc_name
                    break
                if (next_task.forSlState == 1):
                    #if (not self.networkQueue.empty()):
                    self.set_ddpg_weights()
                    #print('setting weights done from ddpg')
                else:
                    #run SLTask
                    state = next_task()
                    #predict
                    #create input vector
                    # print("creating input vector for predict")
                    s = [state.mPos,state.mSpd,state.gAng,state.gSpd]
                    s = generalFunctions.scaleState(s)
                    state.pwr = sess.run(scaled_out,feed_dict={inputs: np.reshape(s,(1,4))}) 
                    print('pwr from SL: ',state.pwr)
                    
                    # print("found new prediction of action/power")
                    self.stateQueue.put(state)                
                                      
                    #Check networkqueue for updated NN values, and stuff it in the tensorflow
                    # update network
                    if (not self.networkQueue.empty()):
                        self.set_ddpg_weights()

                    if (self.debug):
                        print("show state")
                        next_task.state.showMe()

            return