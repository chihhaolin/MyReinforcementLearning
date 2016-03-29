import game.wrapped_flappy_bird as game    
import numpy as np
import pickle
import time
import random
import cv2
from collections import deque
import lasagne
import theano
import theano.tensor as T


MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 50000  # size of D
# AGENT_HISTORY_LENGTH (fixed 4):  The number of most recent frames experienced by the agent that are given as input to the Q network
TARGET_NETWORK_UPDATE_FREQ = 10000  # The frequency with which the target netwrok is updated ( parameter C from Algorithm 1)


DISCOUNT_FACTOR = 0.99 # Discount factor gamma used in the Q-learning update. 
# ACTION_REPEAT
# UPDATE_FREQUENCY
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
SQUARED_GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01

INITIAL_EXPORLATION_EPSILON = 1.0 # initial value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_EPSILON = 0.000001 # final value of epsilon in epsilon-greedy exploration
FINAL_EXPLORATION_FRAME = 500000 #300000  # The number of frames over which the initial value of epsilon is linearly annealed to its final value
REPLAY_START_SIZE = 10000 #100000  ## OBSERVATION:  A uniform random policy is run for this number of frames before training 


ACTIONS = 2 # number of valid actions
K_FRAME_SELECTION = 2  # 60Hz  => K = 4, 30Hz => K = 2


def createNetwork():
    net = {}
    net['input'] = lasagne.layers.InputLayer((None, 4, 84, 84))
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], 32, 8, stride=4)
    net['conv2'] = lasagne.layers.Conv2DLayer(net['conv1'], 64, 4, stride=2)
    net['conv3'] = lasagne.layers.Conv2DLayer(net['conv2'], 64, 3, stride=1)
    net['fc4'] = lasagne.layers.DenseLayer(net['conv3'], num_units=512)
    net['fc5'] = lasagne.layers.DenseLayer(net['fc4'], num_units=2, nonlinearity=None)
    return net


Training_Frame = 50000000   # 50x10^6, 50 million
loading_path = ""
observed_frame = 0

Q_rec = 0
Cost_rec = 0
Epoch = 0
if __name__ == '__main__':

    net = createNetwork()
    ## loading network parameters
    # params = pickle.load(open(loading_path,"rb"))
    # lasagne.layers.set_all_param_values(net['fc5'], params)   
    # print "loading params successfully"
    ####    
    input_X = T.tensor4('input_X')
    target_Y = T.vector("target_Y") 
    action_input = T.matrix("action")

    pred_Y = lasagne.layers.get_output(net['fc5'], inputs=input_X)
    Action_Y_index = T.argmax(pred_Y, axis=1)
    
    error_term = target_Y - T.nonzero_values(action_input * pred_Y)
    cost = T.mean(T.sqr(error_term))

    #scaled_error_term = lasagne.updates.norm_constraint(error_term, max_norm=1, norm_axes=0)
    #cost = T.mean(T.sqr(scaled_error)) 

    params = lasagne.layers.get_all_params(net['fc5'], trainable=True)
    updates = lasagne.updates.adam(cost, params, learning_rate=LEARNING_RATE, beta1=GRADIENT_MOMENTUM, 
                                                 beta2=SQUARED_GRADIENT_MOMENTUM, epsilon=MIN_SQUARED_GRADIENT)
    
    average_Q = T.mean(T.nonzero_values(action_input * pred_Y))
    Q_value = T.max(pred_Y, axis=1)
    
    train_fn = theano.function( inputs=[input_X, action_input, target_Y], updates=updates, outputs=[average_Q, cost]) 
    action_index_fn = theano.function( inputs=[input_X], outputs=[Action_Y_index])
    Q_value_fn = theano.function( inputs=[input_X], outputs=[Q_value])

    debug_fn = theano.function( inputs=[input_X, action_input], outputs=[pred_Y, Action_Y_index,  T.nonzero_values(action_input * pred_Y)]) 

    ## game start !!
    game_state = game.GameState()

    # store the replay memory
    D = deque()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS, dtype=theano.config.floatX)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    x_t = np.expand_dims(x_t, axis=0)
    s_t = np.vstack((x_t, x_t, x_t, x_t))

    ## debug
    # a_t = np.zeros(ACTIONS, dtype=theano.config.floatX)
    # a_t[1] = 1    
    # aaa = debug_fn([s_t], [a_t])
    # print "pred_Y:", aaa[0]
    # print "action Y index", aaa[1]
    # print "select pred_Y", aaa[2]

    # a_t = np.zeros(ACTIONS, dtype=theano.config.floatX)
    # a_t[0] = 1    
    # aaa = debug_fn([s_t], [a_t])
    # print "pred_Y:", aaa[0]
    # print "action Y index", aaa[1]
    # print "select pred_Y", aaa[2]


    ## initialization epsilon
    if observed_frame > FINAL_EXPLORATION_FRAME:
        epsilon = FINAL_EXPLORATION_EPSILON
    else:
        epsilon = INITIAL_EXPORLATION_EPSILON - observed_frame*((INITIAL_EXPORLATION_EPSILON - FINAL_EXPLORATION_EPSILON)/FINAL_EXPLORATION_FRAME)

    ## training
    a_tm1 = do_nothing
    for t in xrange(Training_Frame-observed_frame):

        ## frame-skipping technique
        if t % K_FRAME_SELECTION == 0:
            a_t = np.zeros(ACTIONS, dtype=theano.config.floatX)
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)  # random action
            else:
                action_index = action_index_fn([s_t]) # theano function
            a_t[action_index] = 1
            a_tm1 = a_t
        else:
            a_t = a_tm1
            game_state.frame_step(a_t) 
            continue 

        ## linearly annealed epsilon
        if epsilon > FINAL_EXPLORATION_EPSILON:
            epsilon -= K_FRAME_SELECTION*(INITIAL_EXPORLATION_EPSILON - FINAL_EXPLORATION_EPSILON) / FINAL_EXPLORATION_FRAME        

        ## take action
        x_t1, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1, (84, 84)), cv2.COLOR_BGR2GRAY)
        x_t1 = np.expand_dims(x_t1, axis=0)
        s_t1 = np.append(x_t1, s_t[:3], axis = 0)

        ## store transition (s_t, a_t, r_t, s_t1, terminal) in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        # update the old values
        s_t = s_t1

        if len(D) > REPLAY_MEMORY_SIZE:
            D.popleft()
        
        ## start training condition: t > REPLAY_START_SIZE (OBSERVATION)
        # Q_rec = None
        # Epoch = 0
        if t > REPLAY_START_SIZE and (t % TARGET_NETWORK_UPDATE_FREQ == 0):
            print "training network..."
            # sample a minibatch of transitions from D
            #_iteration = len(D) / MINIBATCH_SIZE
            _iteration = TARGET_NETWORK_UPDATE_FREQ / MINIBATCH_SIZE
            Q_rec = 0
            Cost_rec = 0
            for _ in xrange(_iteration):
                minibatch = random.sample(D, MINIBATCH_SIZE)

                s_j_batch = [d[0] for d in minibatch]
                a_j_batch = [d[1] for d in minibatch]
                r_j_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_j_batch = []

                Q_value_s_j1_batch = Q_value_fn(s_j1_batch)  # theano function
                Q_value_s_j1_batch = Q_value_s_j1_batch[0]

                for i in range(0, len(minibatch)):
                    # if terminal, only equals reward
                    if minibatch[i][4]:
                        y_j_batch.append(r_j_batch[i])
                    else:
                        y_j_batch.append(r_j_batch[i] + DISCOUNT_FACTOR * Q_value_s_j1_batch[i])

                Q, _cost = train_fn(s_j_batch, a_j_batch ,np.array(y_j_batch,dtype=theano.config.floatX))            
                Q_rec += Q 
                Cost_rec += _cost

            Q_rec /= _iteration
            Cost_rec /= _iteration
            Epoch += 1

            print "ending training network..."
            print
            print "Training Frame:", t + observed_frame
            print "Training Epoch:", Epoch
            print "Average Q:", Q_rec
            print "Average Cost:", Cost_rec
            print "Epsilon", epsilon


        # save progress every 10000 iterations
        if (t % 10000 == 0) and (t > 0):
            params = lasagne.layers.get_all_param_values(net['fc5'])   
            path = "networks_training_frame" +  str(t + observed_frame) + ".pkl"
            pickle.dump(params,  open(path, "wb"))
            

