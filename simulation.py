"""
This documents uses code from Patrick Emami's DDPG implementation, found here:
https://github.com/pemami4911/deep-rl/tree/master/ddpg
And the Open AI gym enviroment, from Open AI:
https://github.com/openai/gym
Adapted by Rasmus Loft and Andreas Danebo Jensen
"""


""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import copy
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

from replay_buffer import ReplayBuffer

class CartPoleEnvCont(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(2) #not used since switch to continuous action space
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if (np.isnan(action)|np.isinf(action)):
            sys.exit('nan or inf action')
#        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #print(action)
        state = self.state
        x, x_dot, theta, theta_dot = state
        #force = self.force_mag if action==1 else -self.force_mag
        force = self.force_mag * action
        #print('action',action)
        if (force > 10):
            force = 10
        if (force < -10):
            force = -10            
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            #reward = 1.
            #modified reward from pendulum. punishes reward for angle, a-speed, action taken, and distance from center
            reward =  -(pow(theta,2) + 0.1*pow(theta_dot,2) + 0.001*pow(action,2)+ 0.1*pow(abs(x),2))
            if type(reward) is np.ndarray:      
                reward = reward[0]
            else:
                reward = reward
            #print ('reward',reward)
            
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 2.0-(pow(theta,2) + 0.1*pow(theta_dot,2) + 0.001*pow(action,2)+ 0.1*pow(abs(x),2))
            if type(reward) is np.ndarray:    
                reward = reward[0]
            else:
                reward = reward
            #if (x < -self.x_threshold or x > self.x_threshold):
            #    reward = -10.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')



# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 200
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        self.net = tflearn.fully_connected(inputs, 400, activation='relu', name="a1")
        self.net2 = tflearn.fully_connected(self.net, 300, activation='relu', name="a2")
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            self.net2, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400, activation='relu')

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================


def train(sess, env, actor, critic):
    global RENDER_ENV
    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    
    count =0
    a2b3 = None

    for i in range(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0
        count = count+1
        for j in range(MAX_EP_STEPS):

            if (RENDER_ENV):
                env.render()

            # Added exploration noise
            if (count>50):
                RENDER_ENV = True
                a = actor.predict(np.reshape(s, (1, 4))) # + (1. / (1. + i))

            if (count>55):
                RENDER_ENV = False
                count =0

            # use PID for actions
            if (count<=50):
            #else:
                parameters = [0.14075554,  0.59973658,  2.1975197 ,  1.37863025]
                a = np.matmul(parameters,s)
                a = a*0.5+ (np.random.rand(1)*2-1)*0.5
                a = np.array([a])

            s2, r, terminal, info = env.step(a[0])
            if (j ==MAX_EP_STEPS-1):
                terminal = True

            replay_buffer.add(np.reshape(s, (4,)), np.reshape(a, (1,)), r,
                              terminal, np.reshape(s2, (4,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]: # is terminal state
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
                
                # update multi process items:
                # a1 =  [v for v in tf.trainable_variables() if v.name == "a1/W:0"][0]
                # a1b = [v for v in tf.trainable_variables() if v.name == "a1/b:0"][0]
                # a2 =  [v for v in tf.trainable_variables() if v.name == "a2/W:0"][0]
                # a2b = [v for v in tf.trainable_variables() if v.name == "a2/b:0"][0]
                # a2b = [v for v in tf.trainable_variables() if v.name == "a2/b:0"][0]

                # a2bB = sess.run(a2b)
                #print('a2bB',a2bB)
                #a2bB = a2bB *2.0                

                # model = tflearn.DNN(actor.out)
                # model.set_weights(actor.net2.b, a2bB)
                # a2b2 = [v for v in tf.trainable_variables() if v.name == "a2/b:0"][0]
                # a2bB2 = sess.run(a2b2)

                #a2b = actor.network_params#[v for v in actor.network_params if v.name == "a2/b:0"][0]
                #a2b = sess.run(a2b)
                #a2b = a2b * 2.0
                #print('a2b',a2b)
                # a2b3 = [v for v in tf.trainable_variables() if v.name == "a2/b:0"][0]
                # a2b3 = sess.run(a2b3)
                
                #if (a2b3 is None):
                # a2b3 = actor.network_params#[v for v in actor.network_params if v.name == "a2/b:0"][0]
                # a2b3 = sess.run(a2b3)
                # a2b3[0][3][3] =0
                    #print(a2b3[0][3][3])
                #update_network_params = \
                #    actor.network_params[0][3].assign(0)
                    #var =tf.Variable(0.);
                    #print (var)
                #sess.run(actor.network_params.assign(actor.network_params))
                
                # var2 = \
                # [actor.network_params[i].assign(a2b3[i])
                # for i in range(len(actor.network_params))]
                # sess.run(var2)
                # a2b3 = actor.network_params#[v for v in actor.network_params if v.name == "a2/b:0"][0]
                # #print(a2b3)
                # a2b3 = sess.run(a2b3)
                # print((a2b3[0][3][3]))
                
                # [<tf.Variable 'a1/W:0' shape=(4, 400) dtype=float32_ref>,
                # <tf.Variable 'a1/b:0' shape=(400,) dtype=float32_ref>, 
                # <tf.Variable 'a2/W:0' shape=(400, 300) dtype=float32_ref>, 
                # <tf.Variable 'a2/b:0' shape=(300,) dtype=float32_ref>, 
                # <tf.Variable 'FullyConnected/W:0' shape=(300, 1) dtype=float32_ref>, 
                # <tf.Variable 'FullyConnected/b:0' shape=(1,) dtype=float32_ref>]
                
                
                #    update_network_params = \
                #    [actor.network_params[i].assign(a2b3[i]) for i in range(len(actor.network_params))]

                
                
                    
                #print(actor.network_params)


                
                # add all items from queue to RB:
                #replay_buffer.add(np.reshape(s, (4,)), np.reshape(a, (1,)), r,
                #              terminal, np.reshape(s2, (4,)))
             

            s = s2
            ep_reward = ep_reward + r
            


            if terminal:

                # summary_str = sess.run(summary_ops, feed_dict={
                    # summary_vars[0]: ep_reward,
                    # summary_vars[1]: ep_ave_max_q / float(j)
                # })

                # writer.add_summary(summary_str, i)
                # writer.flush()

                print ('| Reward: %.2f' % (ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j)))

                break


def main(_):
    with tf.Session() as sess:

        saver = tf.train.Saver()
        #env = gym.make(ENV_NAME)
        env = CartPoleEnvCont()
        env.render()
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = 1# env.action_space.shape[0]
        #print('action_dim',action_dim)
        action_bound = [ 1.] # env.action_space.high
        #print('action_bound',action_bound)
        #action_dim 1
        #action_bound [ 2.]
        
        
        # Ensure action bound is symmetric
        #assert (env.action_space.high == -env.action_space.low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = wrappers.Monitor(
                    env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, MONITOR_DIR, force=True)

        train(sess, env, actor, critic)

        if GYM_MONITOR_EN:
            env.monitor.close()

if __name__ == '__main__':
    tf.app.run()
