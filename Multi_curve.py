#!/usr/bin/env python
# Time: 2021/1/4 下午10:15
# Author: Yichuan


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# useful function for smoothing curves
def smooth(y, radius, mode='two_sided', valid_only=False):
    """
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':[Note that twosided mode has better performance in smoothing this cure]
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    """
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()  # np.ones_like outputs an array that has the same dimension but just 1s
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)  # np.ones outputs an array looks like [1,1,1...]
        out = np.convolve(y, convkernel, mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    else:
        # 'causal'
        convkernel = np.ones(radius)  # radius stands for the width of the sliding window
        out = np.convolve(y, convkernel, mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


sns.set()  # 因为sns.set()一般不用改，可以在导入模块时顺便设置好
"style must be one of white, dark, whitegrid, darkgrid, ticks"
sns.set_theme(style="whitegrid")
length = 1000
# 1111111111111111111111111111111111111111111111
#######################################################################################################################
rewards0 = []
with open("./NET_1/data_0.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards0.append(float(log.split(" ")[-1]))
rewards0 = smooth(rewards0[0:length], radius=10)

rewards1 = []
with open("./NET_1/data_1.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards1.append(float(log.split(" ")[-1]))
rewards1 = smooth(rewards1[0:length], radius=10)

rewards2 = []
with open("./NET_1/data_2.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards2.append(float(log.split(" ")[-1]))
rewards2 = smooth(rewards2[0:length], radius=10)

rewards3 = []
with open("./NET_1/data_3.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards3.append(float(log.split(" ")[-1]))
rewards3 = smooth(rewards3[0:length], radius=10)

rewards4 = []
with open("./NET_1/data_4.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards4.append(float(log.split(" ")[-1]))
rewards4 = smooth(rewards4[0:length], radius=10)

rewards = np.concatenate(
    (rewards0[0:length], rewards1[0:length], rewards2[0:length], rewards3[0:length], rewards4[0:length]))  # 合并数组

####################################################################################################################
episode0 = range(length)
episode1 = range(length)
episode2 = range(length)
episode3 = range(length)
episode4 = range(length)
episode = np.concatenate((episode0, episode1, episode2, episode3, episode4))
###################################################################################################################

# 2222222222222222222222222222222222222
#######################################################################################################################
rewards00 = []
with open("./NET_3/data_0.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards00.append(float(log.split(" ")[-1]))
rewards00 = smooth(rewards00[0:length], radius=10)

rewards11 = []
with open("./NET_3/data_1.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards11.append(float(log.split(" ")[-1]))
rewards11 = smooth(rewards11[0:length], radius=10)

rewards22 = []
with open("./NET_3/data_2.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards22.append(float(log.split(" ")[-1]))
rewards22 = smooth(rewards22[0:length], radius=10)

rewards33 = []
with open("./NET_3/data_3.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards33.append(float(log.split(" ")[-1]))
rewards33 = smooth(rewards33[0:length], radius=10)

rewards44 = []
with open("./NET_3/data_4.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards44.append(float(log.split(" ")[-1]))
rewards44 = smooth(rewards44[0:length], radius=10)

Rewards = np.concatenate(
    (rewards00[0:length], rewards11[0:length], rewards22[0:length], rewards33[0:length], rewards44[0:length]))  # 合并数组

####################################################################################################################
episode00 = range(length)
episode11 = range(length)
episode22 = range(length)
episode33 = range(length)
episode44 = range(length)
Episode = np.concatenate((episode00, episode11, episode22, episode33, episode44))
###################################################################################################################

# 33333333333333333333333333333333333333333333
#######################################################################################################################
rewards000 = []
with open("./PPO/data_0.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards000.append(float(log.split(" ")[-1]))
rewards000 = smooth(rewards000[0:length], radius=10)

rewards111 = []
with open("./PPO/data_1.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards111.append(float(log.split(" ")[-1]))
rewards111 = smooth(rewards111[0:length], radius=10)

rewards222 = []
with open("./PPO/data_2.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards222.append(float(log.split(" ")[-1]))
rewards222 = smooth(rewards222[0:length], radius=10)

rewards333 = []
with open("./PPO/data_3.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards333.append(float(log.split(" ")[-1]))
rewards333 = smooth(rewards333[0:length], radius=10)

rewards444 = []
with open("./PPO/data_4.txt") as file:
    logs = file.readlines()
    for log in logs:
        rewards444.append(float(log.split(" ")[-1]))
rewards444 = smooth(rewards444[0:length], radius=10)

RRewards = np.concatenate((rewards000[0:length], rewards111[0:length], rewards222[0:length], rewards333[0:length],
                           rewards444[0:length]))  # 合并数组

####################################################################################################################
episode000 = range(length)
episode111 = range(length)
episode222 = range(length)
episode333 = range(length)
episode444 = range(length)
EEpisode = np.concatenate((episode000, episode111, episode222, episode333, episode444))
###################################################################################################################


sns.lineplot(x=episode, y=rewards)
sns.lineplot(x=Episode, y=Rewards)
sns.lineplot(x=EEpisode, y=RRewards)

plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.ylim(0, 210)
plt.xlim(0, length)
plt.legend(['BNPPO-switch', 'BNPPO-concat', 'PPO'])

plt.savefig("CartPole Data Efficiency.pdf")

plt.show()
