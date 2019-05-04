import numpy as np
import matplotlib.pyplot as plt
import time

class Simulator:

    def __init__(self, dt=0.01, n_state=100, xmin=-1, xmax=2, vmin=-2,
                 vmax=2, state_obj=1 ):

        self.ns = n_state
        self.xmin = xmin
        self.xmax = xmax
        self.vmin = vmin
        self.vmax = vmax
        self.sobj = state_obj
        self.dt = dt
        self.X = np.array((0, 0), dtype=np.float64)

    def act(self, action):
        self.X += self.dt*np.array((self.X[1], action))
        reward = -abs(self.X[0]-self.sobj)
        next_state = self.observe()
        return reward, next_state

    def observe(self):

        if self.X[0] < self.xmin:
            x = 0
        elif self.X[0] > self.xmax:
            x = self.ns-1
        else:
            x = int( (self.X[0]-self.xmin)/(self.xmax-self.xmin)*(self.ns-2) + 1 )

        if self.X[1] < self.vmin:
            v = 0
        elif self.X[1] > self.vmax:
            v = self.ns-1
        else:
            v = int( (self.X[1]-self.vmin)/(self.vmax-self.vmin)*(self.ns-2) + 1 )

        return (x, v)

class StateActionValue:

    def __init__(self, n_state=100, n_action=100):
        self.ns = n_state
        self.na = n_action
        self.Q = -100*np.ones((n_state, n_state, n_action))

    def policy(self, state, eps):
        if np.random.rand() > eps:
            return np.argmax(self.Q[state[0], state[1], :])
        else:
            return np.random.randint(na)

    def policyMap(self):
        return np.array( [[ self.policy((i,j), 0) for i in range(self.ns) ] \
                                                  for j in range(self.ns) ] )

    def V(self, state):
        return np.amax(self.Q[state[0], state[1], :])

    def VMap(self):
        return np.array( [[ self.V((i,j)) for i in range(self.ns) ] \
                                          for j in range(self.ns) ] )

    def update(self, state, action_idx, next_state, reward, alpha):
        self.Q[state[0], state[1], action_idx] += \
            alpha*(reward + self.V(next_state) - self.V(state))


ns = 100
sobj = 66

na = 10
amin = -1
amax = 1

eps0 = 0.01
alpha0 = 0.01
n_episodes = int(1e6)

n_steps = 50
dt = 0.05

actions = np.linspace(amin, amax, na)
Q = StateActionValue(n_state=ns, n_action=na)

fig1, ax1 = plt.subplots()
im1 = ax1.imshow(Q.policyMap(), interpolation='none')
fig1.show()
fig2, ax2 = plt.subplots()
im2 = ax2.imshow(Q.VMap(), interpolation='none')
fig2.show()
fig3, ax3 = plt.subplots()
ln3, = ax3.plot([], [])
ax3.set_xlim((-1, 2))
ax3.set_ylim((-2, 2))
fig3.show()
fig4, ax4 = plt.subplots()
ln4, = ax4.plot([], [])
ax4.set_xlim((0, dt*n_steps))
ax4.set_ylim((-1, 1))
fig4.show()

#time1 = time.time()

for i in range(n_episodes):
    sim = Simulator(n_state=ns, dt=dt)
    state = sim.observe()
    tot_reward = 0
    for j in range(n_steps):
        eps = eps0
        alpha = alpha0
        action_idx = int(Q.policy(state, eps))
        action = actions[action_idx]
        reward, next_state = sim.act(action)
        Q.update(state, action_idx, next_state, reward, alpha)
        state = next_state
        tot_reward += reward

    if i%10000 == 0:
        #time2 = time.time()
        #print('episode {} -- tot_reward={} -- computation time='.format(i, tot_reward, time2-time1))

        sim = Simulator(n_state=ns, dt=dt)
        state = sim.observe()
        state_list = np.empty((n_steps, 2))
        action_list = np.empty(n_steps)
        tot_reward = 0
        for j in range(n_steps):
            action_idx = int(Q.policy(state, 0))
            action = actions[action_idx]
            reward, next_state = sim.act(action)
            state = next_state
            state_list[j,:] = sim.X
            action_list[j] = action
            tot_reward += reward
        ln3.set_data(state_list[:,0], state_list[:,1])
        fig3.canvas.draw()
        t = dt*np.linspace(1, n_steps, n_steps)
        ln4.set_data(t, action_list)
        fig4.canvas.draw()

        print('episode {} -- tot_reward={}'.format(i, tot_reward))
        im1 = ax1.imshow(Q.policyMap(), interpolation='none')
        fig1.canvas.draw()
        im2 = ax2.imshow(Q.VMap(), interpolation='none')
        fig2.canvas.draw()


plt.show()
