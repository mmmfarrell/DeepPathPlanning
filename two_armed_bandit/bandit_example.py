import tensorflow as tf
import numpy as np

# Define our bandits
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)

def pull_bandit(bandit):
    # get a random number
    rand = np.random.randn(1)
    if rand > bandit:
        # return a positve reward
        return 1
    else:
        return -1

tf.reset_default_graph()

# Define graph
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# Define training method
# Two placeholders for reward and action.
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)

# Grab the weight responsible for our action
responsible_weight = tf.slice(weights, action_holder, [1])

# Compute policy loss equation.
loss = -(tf.log(responsible_weight)*reward_holder)

# Optimizer to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
update = optimizer.minimize(loss)

# Training the agent
total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.5

init = tf.initialize_all_variables()

# Launch graph and train
with tf.Session() as sess:
    sess.run(init)
    for i in range(total_episodes):
        # Choose random action or one from network
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)

        # Get our reward from the bandit we choose.
        reward = pull_bandit(bandits[action])

        # Update network
        _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={reward_holder:[reward], action_holder:[action]})

        # Update our running tally of scores.
        total_reward[action] += reward
        
        # Print running totals every 50 iterations.
        if i % 50 == 0:
            print "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward)

# Print our guess after training has concluded.
print "\nThe agent thinks bandit " + str(np.argmax(ww) + 1) + " is the most promising."
print "Correct answer was bandit # " + str(np.argmin(np.array(bandits)) + 1)

