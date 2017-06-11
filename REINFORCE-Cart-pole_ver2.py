"""
This class implements the reinforce algorithm on the Open AI Gym 
Cartpole code reused from Open AI Gym
Adapted by Rasmus Loft and Andreas Danebo Jensen
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

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
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
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
#        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        #print(action)
        state = self.state
        x, x_dot, theta, theta_dot = state
        #force = self.force_mag if action==1 else -self.force_mag
        force = self.force_mag * action
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
            # reward = 1.0
            reward =  -(pow(theta,2) + 0.1*pow(theta_dot,2) + 0.001*pow(action,2)+ 0.1*pow(abs(x),2))
            if type(reward) is np.ndarray:      
                reward = 2+reward[0]
            else:
                reward = 2+reward
            #print ('reward',reward)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # reward = 1.0
            reward =  -(pow(theta,2) + 0.1*pow(theta_dot,2) + 0.001*pow(action,2)+ 0.1*pow(abs(x),2))
            if type(reward) is np.ndarray:      
                reward = 2+reward[0]
            else:
                reward = 2+reward
            #print ('reward',reward)
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
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

import gym
import itertools
import matplotlib
import numpy as np
import sys
import collections
from pathlib import Path

if "../" not in sys.path:
  sys.path.append("../") 



env = CartPoleEnvCont() #gym.envs.make("CartPole-v1")

class PolicyEstimator():
    """
    Policy Function approximator. 
    """
    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        self.learning_rate = learning_rate
        phi_file = Path("./phi.csv")
        # print(phi_file)
        if (phi_file.is_file()):
            self.phiTable = np.loadtxt("phi.csv",delimiter=";")
            # self.phi = self.phiTable[-1] #using the phi weights of the previous run
            self.phi = np.array([0.5,0.5,0.5,0.5]) #using standard phi weights
            # self.phi = np.random.random_sample((4,)) #using random phi weights
        else:    
            self.phi = np.array([0.5,0.5,0.5,0.5]) #np.random.random_sample((4,))
        print("phi = ",self.phi)
        self.variance = 0.1
        self.phiTable = []
        self.phiTable.append(self.phi)
        self.actionTable = []
        self.learning = True
    
    def predict(self, state, sess=None):
        #print("state:",state)
        #print("phi:",self.phi)
        if (self.learning == True):
            prediction = np.random.normal(np.matmul(state,self.phi),self.variance)
        else:
            prediction = np.matmul(state,self.phi)
        if (prediction < -1):
            prediction = -1
        if(prediction > 1):
            prediction = 1
        self.actionTable.append(prediction)
        return  prediction

    def update(self, state, action, reward):
        self.phiTable.append(self.phi)
        if (self.learning == True):
            upfirst = np.float64(np.matmul(state,self.phi))
            upper = np.array((action - upfirst)*state,dtype="float64")
            downer = np.float64((np.power(self.variance,2)))
            score = np.float64(upper / downer) #np.longdouble((action - np.matmul(state,self.phi))*state/(np.power(self.variance,2)))
            #print("score:",score)
            self.phi = self.phi + self.learning_rate*score*reward 
        return

    def setLearningRate(self):
        self.learning_rate = self.learning_rate*0.9
    
    def decreaseVariance(self):
        if (self.variance <= 0.0125):
            self.variance = 0.0125
            self.setLearningRate()
        else:
            print("Variance has been decreased: ",self.variance)
            self.variance = self.variance*0.9
 
    def increaseVariance(self):
        if (self.variance < 0.1):
            print("Variance has been increased: ",self.variance)
            self.variance = self.variance*1.1 

    def stopLearning(self):
        self.learning = False

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy 
    function approximator using policy gradient.
    
    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized 
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
    
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    def finGeoSer(x,n):
        # Finite Geometric Series: 1 + x + x^2 + ... + x^n
        return (1-np.power(x,n+1))/(1-x)
    
    # Keeps track of useful statistics
    rewards =  [0.]*num_episodes
    lengths = [0.]*num_episodes
    
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    
    Max_iterations = 200
    count_max_reward = 0
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        
        episode = []
        
        # One step in the environment
        for t in range(Max_iterations):#itertools.count():
            
            
            done = False
            # Take a step
            action = estimator_policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            # if (i_episode >= num_episodes - 5):
                # env.render()
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            
            # Update statistics
            rewards[i_episode] += reward
            lengths[i_episode] = t
            
            # Calculate TD Target
            #value_next = estimator_value.predict(next_state)
            #td_target = reward + discount_factor * value_next
            #td_error = td_target - estimator_value.predict(state)
            
            # Update the value estimator
            #estimator_value.update(state, td_target)
            
            # Update the policy estimator
            # using the td error as our advantage estimate
            #estimator_policy.update(state, td_error, action)
            
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, rewards[i_episode]), end="")

            if done:
                break
                
            state = next_state
        
        G = rewards[i_episode]
        episodes = len(episode) #num_episodes
        #print("G = ",G)
        lam = 0.97
        m = finGeoSer(lam,episodes)#0

        Rewt0 = G/m
        for k in range(0,episodes):
            rew = Rewt0 * lam**(k)
            estimator_policy.update(episode[k].state, episode[k].action, rew)
        #print("Phi = ",estimator_policy.phi)   
        if (G >= Max_iterations * 1.95 ):
            print("\nHit reward ceiling")
            count_max_reward += 1
            if (count_max_reward == num_episodes*0.05):
                policy_estimator.stopLearning()

    return [rewards, lengths]


import time
start = int(round(time.time()))
policy_estimator = PolicyEstimator(learning_rate=0.001)
stats = actor_critic(env, policy_estimator, value_estimator, 1000, discount_factor=0.95)
end = int(round(time.time()))

totIt = 0

for row in stats[1]:
    totIt += row
    
print("\rAlgorithm ended after {} seconds and {} iterations".format(
                    (end-start), totIt, end=""))
    
np.savetxt("rew.csv", stats[0], delimiter=";",header="Rewards")

np.savetxt("phi.csv", policy_estimator.phiTable, delimiter=";",header="x;x_dot;theta;theta_dot")

np.savetxt("actions.csv", policy_estimator.actionTable, delimiter=";",header="actions")

