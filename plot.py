import matplotlib.pyplot as plt

# read file from reward.txt
rewards = [] 
with open('reward.txt', 'r') as f:
    lines = f.readlines()
    for line in lines: 
        line = line.strip()
        rewards.append(float(line))

plt.plot(rewards)
plt.ylabel('Reward')
plt.xlabel('Iteration')
plt.show()