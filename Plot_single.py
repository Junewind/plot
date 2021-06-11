import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set()
sns.set_theme(style="whitegrid")

length = 0
rewards = []
with open("Record_y_2000.txt") as file:
    logs = file.readlines()
    for log in logs:
        length += 1
        rewards.append(float(log.split(" ")[-1][:-3]))

# episode = range(length)
episode = np.linspace(0, 50, 11)

plt.plot(episode, rewards)
# plt.xlim(0.03, 0.24)
# plt.ylim(0, 0.5)
plt.xlabel("param", fontdict={'weight': 'normal', 'style': 'italic', 'size': 13})   # 解决坐标轴x标题字体
plt.ylabel("SOI", fontdict={'weight': 'normal', 'style': 'italic', 'size': 13})   # 解决坐标轴y标题字体
plt.legend(['Record_y_2000'], fontsize=13, bbox_to_anchor=(0.55, -0.2))   # 解决图例字号的问题    # plt.legend(fontsize=16, loc='upper right')
plt.tick_params(labelsize=16)   # 解决坐标轴数字字号的问题

# plt.title("BN_PPO FlappyBird")
plt.savefig("Record_y_2000.png", bbox_inches = 'tight')   # 解决保存图片不完整的问题
plt.show()



# 坐标轴刻度的字体大小
# plt.xticks([0, 100, 200, 300, 400, 500, 600, 700])
# plt.tick_params(labelsize=13) #刻度字体大小13