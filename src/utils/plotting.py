import matplotlib.pyplot as plt

def training_reward(reward_over_episodes, name):
    plt.figure(figsize=(12, 6))
    plt.plot(reward_over_episodes, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(name)
    plt.legend()
    plt.savefig(f"{name}.png")
    plt.show()