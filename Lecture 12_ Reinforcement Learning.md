
# Lecture Notes 12: Reinforcement Learning

[ToC]

Welcome to our introduction to reinforcement learning! Reinforcement Learning is the study of agents and how they learn by trial and error. It formalizes the idea that rewarding or punishing an agent for its behavior makes it more likely to repeat or forego that behavior in the future.

## What Can RL Do?
RL methods have recently enjoyed a wide variety of successes. For example, it’s been used to teach computers to control robots in simulation and real world.
![](https://i.imgur.com/ZrfdbYp.png)

![](https://i.imgur.com/pLQ9v4X.png)


It’s also famously been used to create breakthrough AIs for sophisticated strategy games, most notably Go and Dota, taught computers to play Atari games from raw pixels, and trained simulated robots to follow human instructions.

## Key Concepts and Terminology
Agent-environment interaction loop.
![](https://i.imgur.com/qLuub7q.png)


The main characters of RL are the agent and the environment. The environment is the world that the agent lives in and interacts with. At every step of interaction, the agent sees a observation of the state of the world, and then decides on an action to take. The environment changes when the agent acts on it, but may also change on its own.

The agent also perceives a reward signal from the environment, a number that tells it how good or bad the current world state is. The goal of the agent is to maximize its cumulative reward, called return. Reinforcement learning methods are ways that the agent can learn behaviors to achieve its goal.

To talk more specifically what RL does, we need to introduce additional terminology. We need to talk about

(1) states and observations,
(2) action spaces,
(3) policies,
(4) trajectories,
(5) different formulations of return,
(6) the RL optimization problem,
(7) value functions.
### (1) States and Observations
A state s is a complete description of the state of the world. There is no information about the world which is hidden from the state. An observation $O$ is a partial description of a state, which may omit information.

In deep RL, we almost always represent states and observations by a real-valued vector, matrix, or higher-order tensor. For instance, a visual observation could be represented by the RGB matrix of its pixel values; the state of a robot might be represented by its joint angles and velocities.

When the agent is able to observe the complete state of the environment, we say that the environment is fully observed. When the agent can only see a partial observation, we say that the environment is partially observed.

Reinforcement learning notation sometimes puts the symbol for state, $S$, in places where it would be technically more appropriate to write the symbol for observation, $O$. Specifically, this happens when talking about how the agent decides an action: we often signal in notation that the action is conditioned on the state, when in practice, the action is conditioned on the observation because the agent does not have access to the state when the environment is partially observed.


### (2)Action Spaces
Different environments allow different kinds of actions. The set of all valid actions in a given environment is often called the action space. Some environments, like Atari and Go, have discrete action spaces, where only a finite number of moves are available to the agent. Other environments, like where the agent controls a robot in a physical world, have continuous action spaces. In continuous spaces, actions are real-valued vectors.

This distinction has some quite-profound consequences for methods in deep RL. Some families of algorithms can only be directly applied in one case, and would have to be substantially reworked for the other.

### (3) Policies
A policy is a rule used by an agent to decide what actions to take. It can be deterministic, in which case it is usually denoted by $\mu$:

$$a_t = \mu(s_t),$$

or it may be stochastic, in which case it is usually denoted by $\pi$:

$$a_t \sim \pi(\cdot | s_t).$$

Because the policy is essentially the agent’s brain, it’s not uncommon to substitute the word “policy” for “agent”, eg saying “The policy is trying to maximize reward.”

In deep RL, we deal with parameterized policies: policies whose outputs are computable functions that depend on a set of parameters (eg the weights and biases of a neural network) which we can adjust to change the behavior via some optimization algorithm.
![](https://i.imgur.com/dPXhgz0.png)



#### Deterministic Policies
Example: Deterministic Policies. Here is a code snippet for building a simple deterministic policy for a continuous action space in PyTorch, using the torch.nn package:
```python=
pi_net = nn.Sequential(
              nn.Linear(obs_dim, 64),
              nn.Tanh(),
              nn.Linear(64, 64),
              nn.Tanh(),
              nn.Linear(64, act_dim)
            )
```
This builds a multi-layer perceptron (MLP) network with two hidden layers of size 64 and \tanh activation functions. If obs is a Numpy array containing a batch of observations, pi_net can be used to obtain a batch of actions as follows:
```python=
obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
actions = pi_net(obs_tensor)
```

Don’t worry about it if this neural network stuff is unfamiliar to you—this tutorial will focus on RL, and not on the neural network side of things. So you can skip this example and come back to it later. But we figured that if you already knew, it could be helpful.

#### Stochastic Policies
The two most common kinds of stochastic policies in deep RL are categorical policies and diagonal Gaussian policies.

Categorical policies can be used in discrete action spaces, while diagonal Gaussian policies are used in continuous action spaces.

Two key computations are centrally important for using and training stochastic policies:

sampling actions from the policy,
and computing log likelihoods of particular actions, $\log \pi_{\theta}(a|s)$.
In what follows, we’ll describe how to do these for both categorical and diagonal Gaussian policies.

##### Categorical Policies

A categorical policy is like a classifier over discrete actions. You build the neural network for a categorical policy the same way you would for a classifier: the input is the observation, followed by some number of layers (possibly convolutional or densely-connected, depending on the kind of input), and then you have one final linear layer that gives you logits for each action, followed by a softmax to convert the logits into probabilities.

Sampling. Given the probabilities for each action, frameworks like PyTorch and Tensorflow have built-in tools for sampling. For example, see the documentation for Categorical distributions in PyTorch, torch.multinomial, tf.distributions.Categorical, or tf.multinomial.

Log-Likelihood. Denote the last layer of probabilities as $P_{\theta}(s)$. It is a vector with however many entries as there are actions, so we can treat the actions as indices for the vector. The log likelihood for an action a can then be obtained by indexing into the vector:

$$\log \pi_{\theta}(a|s) = \log \left[P_{\theta}(s)\right]_a$$.

##### Diagonal Gaussian Policies

A multivariate Gaussian distribution (or multivariate normal distribution, if you prefer) is described by a mean vector, $\mu$, and a covariance matrix, $\Sigma$. A diagonal Gaussian distribution is a special case where the covariance matrix only has entries on the diagonal. As a result, we can represent it by a vector.

A diagonal Gaussian policy always has a neural network that maps from observations to mean actions, $\mu_{\theta}(s)$. There are two different ways that the covariance matrix is typically represented.

The first way: There is a single vector of log standard deviations, $\log \sigma$, which is not a function of state: the $\log \sigma$ are standalone parameters. (You Should Know: our implementations of VPG, TRPO, and PPO do it this way.)

The second way: There is a neural network that maps from states to log standard deviations, $\log \sigma_{\theta}(s)$. It may optionally share some layers with the mean network.

Note that in both cases we output log standard deviations instead of standard deviations directly. This is because log stds are free to take on any values in $(-\infty, \infty)$, while stds must be nonnegative. It’s easier to train parameters if you don’t have to enforce those kinds of constraints. The standard deviations can be obtained immediately from the log standard deviations by exponentiating them, so we do not lose anything by representing them this way.

Sampling. Given the mean action $\mu_{\theta}(s)$ and standard deviation $\sigma_{\theta}(s)$, and a vector $z$ of noise from a spherical Gaussian $(z \sim \mathcal{N}(0, I))$, an action sample can be computed with

$$ a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$$

where $\odot$ denotes the elementwise product of two vectors. Standard frameworks have built-in ways to generate the noise vectors, such as torch.normal or tf.random_normal. Alternatively, you can build distribution objects, eg through torch.distributions.Normal or tf.distributions.Normal, and use them to generate samples. (The advantage of the latter approach is that those objects can also calculate log-likelihoods for you.)

Log-Likelihood. The log-likelihood of a k-dimensional action $a$, for a diagonal Gaussian with mean $\mu = \mu_{\theta}(s)$ and standard deviation $\sigma = \sigma_{\theta}(s)$, is given by

$$ \log \pi_{\theta}(a|s) = -\frac{1}{2}\left(\sum_{i=1}^k \left(\frac{(a_i - \mu_i)^2}{\sigma_i^2} + 2 \log \sigma_i \right) + k \log 2\pi \right)$$

### (4) Trajectories
A trajectory $\tau$ is a sequence of states and actions in the world,

$$\tau = (s_0, a_0, s_1, a_1, ...).$$

The very first state of the world, $s_0$, is randomly sampled from the start-state distribution, sometimes denoted by $\rho_0$:

$$s_0 \sim \rho_0(\cdot).$$

State transitions (what happens to the world between the state at time $t$, $s_t$, and the state at $t+1$, $s_{t+1}$), are governed by the natural laws of the environment, and depend on only the most recent action, $a_t$. They can be either deterministic,

$$s_{t+1} = f(s_t, a_t)$$

or stochastic,

$$s_{t+1} \sim P(\cdot|s_t, a_t).$$

Actions come from an agent according to its policy.

Trajectories are also frequently called episodes or rollouts.

### (5) Reward and Return
The reward function $R$ is critically important in reinforcement learning. It depends on the current state of the world, the action just taken, and the next state of the world:

$$r_t = R(s_t, a_t, s_{t+1})$$

although frequently this is simplified to just a dependence on the current state, $r_t = R(s_t)$, or state-action pair $r_t = R(s_t,a_t)$.

The goal of the agent is to maximize some notion of cumulative reward over a trajectory, but this actually can mean a few things. We’ll notate all of these cases with $R(\tau)$, and it will either be clear from context which case we mean, or it won’t matter (because the same equations will apply to all cases).

One kind of return is the finite-horizon undiscounted return, which is just the sum of rewards obtained in a fixed window of steps:

$$R(\tau) = \sum_{t=0}^T r_t.$$

Another kind of return is the infinite-horizon discounted return, which is the sum of all rewards ever obtained by the agent, but discounted by how far off in the future they’re obtained. This formulation of reward includes a discount factor $\gamma \in (0,1)$:

$$R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$$

Why would we ever want a discount factor, though? Don’t we just want to get all rewards? We do, but the discount factor is both intuitively appealing and mathematically convenient. On an intuitive level: cash now is better than cash later. Mathematically: an infinite-horizon sum of rewards may not converge to a finite value, and is hard to deal with in equations. But with a discount factor and under reasonable conditions, the infinite sum converges.


While the line between these two formulations of return are quite stark in RL formalism, deep RL practice tends to blur the line a fair bit—for instance, we frequently set up algorithms to optimize the undiscounted return, but use discount factors in estimating value functions.

### (6) The RL Problem
Whatever the choice of return measure (whether infinite-horizon discounted, or finite-horizon undiscounted), and whatever the choice of policy, the goal in RL is to select a policy which maximizes expected return when the agent acts according to it.

To talk about expected return, we first have to talk about probability distributions over trajectories.

Let’s suppose that both the environment transitions and the policy are stochastic. In this case, the probability of a T-step trajectory is:

$$P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t)$$

The expected return (for whichever measure), denoted by $J(\pi)$, is then:

$$J(\pi) = \int_{\tau} P(\tau|\pi) R(\tau) = E_{\tau\sim \pi}{R(\tau)}$$

The central optimization problem in RL can then be expressed by

$$\pi^* = \arg \max_{\pi} J(\pi)$$

with $\pi^*$ being the optimal policy.

### (7) Value Functions
It’s often useful to know the value of a state, or state-action pair. By value, we mean the expected return if you start in that state or state-action pair, and then act according to a particular policy forever after. Value functions are used, one way or another, in almost every RL algorithm.

There are four main functions of note here.

The On-Policy Value Function, $V^{\pi}(s)$, which gives the expected return if you start in state $s$ and always act according to policy $\pi$:

$$V^{\pi}(s) = E_{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}$$

The On-Policy Action-Value Function, Q^{\pi}(s,a), which gives the expected return if you start in state s, take an arbitrary action a (which may not have come from the policy), and then forever after act according to policy \pi:

$$Q^{\pi}(s,a) = E_{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}$$

The Optimal Value Function, $V^*(s)$, which gives the expected return if you start in state s and always act according to the optimal policy in the environment:

$$V^*(s) = \max_{\pi} E_{\tau \sim \pi}{R(\tau)\left| s_0 = s\right.}$$

The Optimal Action-Value Function, $Q^*(s,a)$, which gives the expected return if you start in state $s$, take an arbitrary action $a$, and then forever after act according to the optimal policy in the environment:

$$Q^*(s,a) = \max_{\pi} E_{\tau \sim \pi}{R(\tau)\left| s_0 = s, a_0 = a\right.}$$



There are two key connections between the value function and the action-value function that come up pretty often:
$$V^{\pi}(s) = E_{a\sim \pi}{Q^{\pi}(s,a)}$$
$$V^*(s) = \max_a Q^* (s,a)$$


#### The Optimal Q-Function and the Optimal Action
There is an important connection between the optimal action-value function $Q^*(s,a)$ and the action selected by the optimal policy. By definition, $Q^*(s,a)$ gives the expected return for starting in state $s$, taking (arbitrary) action a, and then acting according to the optimal policy forever after.

The optimal policy in s will select whichever action maximizes the expected return from starting in s. As a result, if we have $Q^*$, we can directly obtain the optimal action, $a^*(s)$, via

$$a^*(s) = \arg \max_a Q^* (s,a)$$

Note: there may be multiple actions which maximize $Q^*(s,a)$, in which case, all of them are optimal, and the optimal policy may randomly select any of them. But there is always an optimal policy which deterministically selects an action.

#### Bellman Equations
All four of the value functions obey special self-consistency equations called Bellman equations. The basic idea behind the Bellman equations is this:

The value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.
The Bellman equations for the on-policy value functions are

\begin{align*}
V^{\pi}(s) &= E_{a \sim \pi \\ s'\sim P}{r(s,a) + \gamma V^{\pi}(s')}, \\
Q^{\pi}(s,a) &= E_{s'\sim P}{r(s,a) + \gamma E_{a'\sim \pi}{Q^{\pi}(s',a')}},
\end{align*}

where $s' \sim P$ is shorthand for $s' \sim P(\cdot |s,a)$, indicating that the next state $s'$ is sampled from the environment’s transition rules; $a \sim \pi$ is shorthand for $a \sim \pi(\cdot|s)$; and $a' \sim \pi$ is shorthand for $a' \sim \pi(\cdot|s')$.

The Bellman equations for the optimal value functions are

\begin{align*}
V^*(s) &= \max_a E_{s'\sim P}{r(s,a) + \gamma V^*(s')}, \\
Q^*(s,a) &= E_{s'\sim P}{r(s,a) + \gamma \max_{a'} Q^*(s',a')}.
\end{align*}

The crucial difference between the Bellman equations for the on-policy value functions and the optimal value functions, is the absence or presence of the \max over actions. Its inclusion reflects the fact that whenever the agent gets to choose its action, in order to act optimally, it has to pick whichever action leads to the highest value.


The term “Bellman backup” comes up quite frequently in the RL literature. The Bellman backup for a state, or state-action pair, is the right-hand side of the Bellman equation: the reward-plus-next-value.

#### Advantage Functions
Sometimes in RL, we don’t need to describe how good an action is in an absolute sense, but only how much better it is than others on average. That is to say, we want to know the relative advantage of that action. We make this concept precise with the advantage function.

The advantage function $A^{\pi}(s,a)$ corresponding to a policy $\pi$ describes how much better it is to take a specific action a in state s, over randomly selecting an action according to $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after. Mathematically, the advantage function is defined by

$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).$$


We’ll discuss this more later, but the advantage function is crucially important to policy gradient methods.



## Value Fitting
If the state space is continuous or large, it is not possible to use a large memory table to record $V(s)$ for every state. However, like other deep learning methods, we can create a function estimator to approximate it.
![](https://i.imgur.com/4YVcwwA.png)
where y is our target value.
![](https://i.imgur.com/Sudbfcd.png)
There are many ways to establish the target value and we will discuss them in the next few sections.
### Monte-Carlo Method
Monte-Carlo method answers questions by repeat random sampling. For example, we can compute the reward of an episode by sampling actions from a policy.
![](https://i.imgur.com/kdfZQtP.png)
By continuous sampling, we apply supervised learning to train Φ to match targets. For example, the rewards (or penality) from San Francisco to San Diego is computed as -10. Therefore, V(SF) equals -10 in this episode.
![](https://i.imgur.com/IMxfXNZ.png)
Monte-Carlo method is non-biased. We play out the whole episode to find the exact reward of an action sequence. However, the Monte-Carlo method has high variance. If the policy or the model is stochastic, the sampled actions are different and lead to different rewards in different episodes.
A stochastic policy is often used in RL to explore space and to smooth out the rewards for a better gradient behavior. However, for each run, the trajectory is different.
![](https://i.imgur.com/Ss5fwh6.png)

To fit the value function modeled by $\phi$, we collect samples and use supervised learning to train the model.
![](https://i.imgur.com/hoEOlDC.png)

In many RL tasks, a small change in action can lead to a very different result. Such a large variance in rewards can destabilize the training badly.
### Temporal difference TD
Let’s assume that we bootstrap the value function from San Francisco to San Diego as (in a real-life example, we usually initialize V with all zero):
![](https://i.imgur.com/34zj9PB.png)
Instead of the Monte-Carlo method, we can use the temporal difference method TD to compute V. We take k-steps and combine the observed rewards with the V value for the state landed after k-steps. For example, in a 1-step TD lookahead, the V(S) of SF equals the rewards from SF to SJ plus V(SJ).
![](https://i.imgur.com/uGqdhAW.png)
First, we observe the rewards for taking each action and then use a 1-step lookahead to update V. The diagram below demonstrates how V(S) is updated after taking three consecutive actions.
![](https://i.imgur.com/tzHwZ4A.png)
This scheme has a lower variance because we take just one action in finding the target for V. However, at least in the early training, it is highly biased as the V values are not accurate. As we progress, the bias in V will lower.
Here, we observe the reward of action and combine it with the fitted value of the next state to train $\phi$.
![](https://i.imgur.com/i9hHrMg.png)


## Policy Gradient
In this section, we’ll discuss the mathematical foundations of policy optimization algorithms, and connect the material to sample code. We will cover three key results in the theory of policy gradients: the simplest equation describing the gradient of policy performance with respect to policy parameters, a rule which allows us to drop useless terms from that expression, and a rule which allows us to add useful terms to that expression.

In the end, we’ll tie those results together and describe the advantage-based expression for the policy gradient—the version we use in our Vanilla Policy Gradient implementation.

![](https://i.imgur.com/dqHFhGu.png)


### Deriving the Simplest Policy Gradient (!!!Test point!!!)
Here, we consider the case of a stochastic, parameterized policy, $\pi_{\theta}$. We aim to maximize the expected return $J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}{R(\tau)}$. For the purposes of this derivation, we’ll take $R(\tau)$ to give the finite-horizon undiscounted return, but the derivation for the infinite-horizon discounted return setting is almost identical.

We would like to optimize the policy by gradient ascent, eg, $\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}$.

The gradient of policy performance, $\nabla_{\theta} J(\pi_{\theta})$, is called the policy gradient, and algorithms that optimize the policy this way are called policy gradient algorithms. 

To actually use this algorithm, we need an expression for the policy gradient which we can numerically compute. This involves two steps: 1) deriving the analytical gradient of policy performance, which turns out to have the form of an expected value, and then 2) forming a sample estimate of that expected value, which can be computed with data from a finite number of agent-environment interaction steps.

In this subsection, we’ll find the simplest form of that expression. In later subsections, we’ll show how to improve on the simplest form to get the version we actually use in standard policy gradient implementations.

We’ll begin by laying out a few facts which are useful for deriving the analytical gradient.

1. Probability of a Trajectory. The probability of a trajectory $\tau = (s_0, a_0, ..., s_{T+1})$ given that actions come from $\pi_{\theta}$ is
\begin{align*}
P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).
\end{align*}
2. The Log-Derivative Trick. The log-derivative trick is based on a simple rule from calculus: the derivative of \log x with respect to x is 1/x. When rearranged and combined with chain rule, we get:
\begin{align*}
\nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).
\end{align*}
3. Log-Probability of a Trajectory. The log-prob of a trajectory is just
\begin{align*}
\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).
\end{align*}
4. Gradients of Environment Functions. The environment has no dependence on \theta, so gradients of $\rho_0(s_0), P(s_{t+1}|s_t, a_t)$, and $R(\tau)$ are zero.

5. Grad-Log-Prob of a Trajectory. The gradient of the log-prob of a trajectory is thus
\begin{align*}
\nabla_{\theta} \log P(\tau | \theta) = \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).
\end{align*}

Putting it all together, we derive the following:

Derivation for Basic Policy Gradient

\begin{align*}
\nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} E_{\tau \sim \pi_{\theta}}{R(\tau)} & \\
&= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
&= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
&= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
&= _{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
\therefore \nabla_{\theta} J(\pi_{\theta}) &= E_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob}
\end{align*}

This is an expectation, which means that we can estimate it with a sample mean. If we collect a set of trajectories $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$ where each trajectory is obtained by letting the agent act in the environment using the policy $\pi_{\theta}$, the policy gradient can be estimated with
$\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)$, where $|\mathcal{D}|$ is the number of trajectories in $\mathcal{D}$ (here, N).

This last expression is the simplest version of the computable expression we desired. Assuming that we have represented our policy in a way which allows us to calculate $\nabla_{\theta} \log \pi_{\theta}(a|s)$, and if we are able to run the policy in the environment to collect the trajectory dataset, we can compute the policy gradient and take an update step.
![](https://i.imgur.com/8BWFPVU.png)


### Implementing the Simplest Policy Gradient
We give a short PyTorch implementation of this simple version of the policy gradient algorithm 
This section was previously written with a Tensorflow example. The old Tensorflow section can be found here.

#### (1) Making the Policy Network.
 ```python=
# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()
```
This block builds modules and functions for using a feedforward neural network categorical policy. (See the Stochastic Policies section in Part 1 for a refresher.) The output from the logits_net module can be used to construct log-probabilities and probabilities for actions, and the get_action function samples actions based on probabilities computed from the logits. (Note: this particular get_action function assumes that there will only be one obs provided, and therefore only one integer action output. That’s why it uses .item(), which is used to get the contents of a Tensor with only one element.)

A lot of work in this example is getting done by the Categorical object on L36. This is a PyTorch Distribution object that wraps up some mathematical functions associated with probability distributions. In particular, it has a method for sampling from the distribution and a method for computing log probabilities of given samples. Since PyTorch distributions are really useful for RL, check out their documentation to get a feel for how they work.


Friendly reminder! When we talk about a categorical distribution having “logits,” what we mean is that the probabilities for each outcome are given by the Softmax function of the logits. That is, the probability for action j under a categorical distribution with logits $x_j$ is: $p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}$

#### (2) Making the Loss Function.
 ```python=
# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```
In this block, we build a “loss” function for the policy gradient algorithm. When the right data is plugged in, the gradient of this loss is equal to the policy gradient. The right data means a set of (state, action, weight) tuples collected while acting according to the current policy, where the weight for a state-action pair is the return from the episode to which it belongs. (Although as we will show in later subsections, there are other values you can plug in for the weight which also work correctly.)



Even though we describe this as a loss function, it is not a loss function in the typical sense from supervised learning. There are two main differences from standard loss functions.

1. The data distribution depends on the parameters. A loss function is usually defined on a fixed data distribution which is independent of the parameters we aim to optimize. Not so here, where the data must be sampled on the most recent policy.

2. It doesn’t measure performance. A loss function usually evaluates the performance metric that we care about. Here, we care about expected return, J(\pi_{\theta}), but our “loss” function does not approximate this at all, even in expectation. This “loss” function is only useful to us because, when evaluated at the current parameters, with data generated by the current parameters, it has the negative gradient of performance.

But after that first step of gradient descent, there is no more connection to performance. This means that minimizing this “loss” function, for a given batch of data, has no guarantee whatsoever of improving expected return. You can send this loss to -\infty and policy performance could crater; in fact, it usually will. Sometimes a deep RL researcher might describe this outcome as the policy “overfitting” to a batch of data. This is descriptive, but should not be taken literally because it does not refer to generalization error.

We raise this point because it is common for ML practitioners to interpret a loss function as a useful signal during training—”if the loss goes down, all is well.” In policy gradients, this intuition is wrong, and you should only care about average return. The loss function means nothing.


The approach used here to make the logp tensor–calling the log_prob method of a PyTorch Categorical object–may require some modification to work with other kinds of distribution objects.

For example, if you are using a Normal distribution (for a diagonal Gaussian policy), the output from calling policy.log_prob(act) will give you a Tensor containing separate log probabilities for each component of each vector-valued action. That is to say, you put in a Tensor of shape (batch, act_dim), and get out a Tensor of shape (batch, act_dim), when what you need for making an RL loss is a Tensor of shape (batch,). In that case, you would sum up the log probabilities of the action components to get the log probabilities of the actions. That is, you would compute:
 ```python
logp = get_policy(obs).log_prob(act).sum(axis=-1)
```
#### (3) Running One Epoch of Training.
```python=
# for training policy
def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens
```


### Baselines in Policy Gradients
An immediate consequence of the EGLP lemma is that for any function b which only depends on state, $E_{a_t \sim \pi_{\theta}}{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)} = 0$.

This allows us to add or subtract any number of terms like this from our expression for the policy gradient, without changing it in expectation:
$\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)}$.

Any function $b$ used in this way is called a baseline.

The most common choice of baseline is the on-policy value function V^{\pi}(s_t). Recall that this is the average return an agent gets if it starts in state s_t and then acts according to policy \pi for the rest of its life.

Empirically, the choice $b(s_t) = V^{\pi}(s_t)$ has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should “feel” neutral about it.

In practice, $V^{\pi}(s_t)$ cannot be computed exactly, so it has to be approximated. This is usually done with a neural network, $V_{\phi}(s_t)$, which is updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy).

The simplest method for learning $V_{\phi}$, used in most implementations of policy optimization algorithms (including VPG, TRPO, PPO, and A2C), is to minimize a mean-squared-error objective:
$\phi_k = \arg \min_{\phi} E_{s_t, \hat{R}_t \sim \pi_k}{\left( V_{\phi}(s_t) - \hat{R}_t \right)^2}$, where $\pi_k$ is the policy at epoch k. This is done with one or more steps of gradient descent, starting from the previous value parameters $\phi_{k-1}$.

Other Forms of the Policy Gradient

What we have seen so far is that the policy gradient has the general form 
$$\nabla_{\theta} J(\pi_{\theta}) = E_{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t}$$

where $\Phi_t$ could be any of $\Phi_t = R(\tau)$, or $\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})$, or with the baseline
$$\Phi_t = \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)$$

All of these choices lead to the same expected value for the policy gradient, despite having different variances. It turns out that there are two more valid choices of weights $\Phi_t$ which are important to know.

## Q-learning
So how can we learn the Q-value? One of the most popular methods is Q-learning with the following steps:
(1) We sample an action.
(2) We observed the reward and the next state.
(3) We take the action with the highest Q.
![](https://i.imgur.com/h1oKNAu.png)

Then we apply the dynamic programming again to compute the Q-value function iteratively:
![](https://i.imgur.com/TL1n8XV.png)
Here is the algorithm of Q-learning with function fitting. Step 2 below reduces the variance by using Temporal Difference. This also improves the sample efficiency comparing with the Monte Carlo method which takes samples until the end of the episode.
![](https://i.imgur.com/b8iyhMv.png)
### Pseudocode
![](https://i.imgur.com/6TD5I5B.png)


Exploration is very important in RL. Without exploration, you will never know what is better ahead. But if it is overdone, we are wasting time. In Q-learning, we have an exploration policy, like epsilon-greedy, to select the action taken in step 1. We pick the action with the highest $Q$ value but yet we allow a small chance of selecting other random actions. $Q$ is initialized with zero. Hence, there is no specific action standing out in early training. As the training progress, more promising actions are selected and the training shift from exploration to exploitation.

## Deep Q-network DQN
Q-learning is unfortunately not very stable with deep learning. In this section, we will finally put all things together and introduce the DQN which beats the human in playing some of the Atari Games by accessing the image frames only.
![](https://i.imgur.com/9yXjanv.png)

DQN is the poster child for Q-learning using a deep network to approximate $Q$. We use supervised learning to fit the Q-value function. We want to duplicate the success of supervised learning but RL is different. In deep learning, we randomize the input samples so the input class is quite balanced and pretty stable across training batches. In RL, we search better as we explore more. So the input space and actions we searched are constantly changing. In addition, as we know better, we update the target value of Q. That is bad news. Both the input and output are under frequent changes.

This makes it very hard to learn the Q-value approximator. DQN introduces experience replay and target network to slow down the changes so we can learn $Q$ gradually. Experience replay stores the last million of state-action-reward in a replay buffer. We train $Q$ with batches of random samples from this buffer. Therefore, the training samples are randomized and behave closer to the supervised learning in Deep Learning.
In addition, we have two networks for storing the values of Q. One is constantly updated while the second one, the target network, is synchronized from the first network once a while. We use the target network to retrieve the Q value such that the changes for the target value are less volatile. Here is the objective for those interested. $D$ is the replay buffer and $\theta^{-}$ is the target network.
![](https://i.imgur.com/i6js9G1.png)

To make training samples independent of each other, we store the latest samples in a replay buffer and randomize them for training. This makes samples less correlated within a training batch. To avoid a moving target, we also delay the Q-value network updates so the target does not change constantly. This topic deserves a whole new article so we will briefly list the algorithm for reference only.
![](https://i.imgur.com/qFdfg6D.png)
### Pseudocode
![](https://i.imgur.com/7MXLM1o.png)
