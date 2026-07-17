import json

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


with open("evaluation.json") as f:
    data = json.load(f)

episodes = data["episodes"]

rewards = [e["reward"] for e in episodes]
steps = [e["steps"] for e in episodes]
success = [e["success"] for e in episodes]
timeouts = [e["timeout"] for e in episodes]
attempts = [e["attempts"] for e in episodes]

actions = np.array(data["action_counts"])

####################################################
# Reward
####################################################

plt.figure(figsize=(8,4))

plt.plot(rewards)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward")

plt.grid(True)

plt.tight_layout()

plt.savefig("reward_curve.png")

####################################################
# Episode Length
####################################################

plt.figure(figsize=(8,4))

plt.plot(steps)

plt.xlabel("Episode")
plt.ylabel("Episode Length")

plt.title("Episode Length per Episode")

plt.grid(True)

plt.tight_layout()

plt.savefig("episode_lengths.png")

####################################################
# Rolling Success Rate
####################################################

window = 20

rolling = []

for i in range(len(success)):
    start = max(0, i - window + 1)
    rolling.append(
        np.mean(success[start:i+1])
    )

plt.figure(figsize=(8,4))

plt.plot(np.array(rolling) * 100)

plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.title("Rolling Success Rate (20 Episodes)")

plt.grid(True)

plt.tight_layout()

plt.savefig("rolling_success.png")


####################################################
# Action Distribution
####################################################

labels = [
    "KEEP",
    "LEFT",
    "RIGHT",
]

percent = (
    actions / actions.sum()
) * 100

plt.figure(figsize=(6,4))

plt.bar(
    labels,
    percent,
)

plt.ylabel("Percentage (%)")

plt.title("Action Distribution")

plt.tight_layout()

plt.savefig("actions.png")


####################################################
# Reward Distribution
####################################################

plt.figure(figsize=(8,4))

plt.hist(
    rewards,
    bins=30,
)

plt.xlabel("Reward")
plt.ylabel("Frequency")

plt.title("Reward Distribution")

plt.grid(True)

plt.tight_layout()

plt.savefig("reward_histogram.png")


####################################################
# Attempts
####################################################

plt.figure(figsize=(8,4))

plt.plot(
    attempts,
    linewidth=1,
)

plt.axhline(
    np.mean(attempts),
    linestyle="--",
    label=f"Mean = {np.mean(attempts):.1f}",
)

plt.legend()

plt.xlabel("Episode")

plt.ylabel("Attempts")

plt.title("Lane Change Attempts")

plt.grid(True)

plt.tight_layout()

plt.savefig("attempts.png")

####################################################
# Success vs Timeout
####################################################

successes = sum(success)
fails = sum(timeouts)

plt.figure(figsize=(6,4))

plt.bar(
    ["Success","Timeout"],
    [successes,fails],
)

plt.ylabel("Episodes")

plt.title("Episode Outcomes")

plt.tight_layout()

plt.savefig("outcomes.png")


####################################################
# Rolling Reward
####################################################

window = 20

rolling_reward = []

for i in range(len(rewards)):
    start = max(0, i-window+1)
    rolling_reward.append(
        np.mean(rewards[start:i+1])
    )

plt.figure(figsize=(8,4))

plt.plot(rolling_reward)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title("Rolling Average Reward")

plt.grid(True)

plt.tight_layout()

plt.savefig("rolling_reward.png")


####################################################
# Cumulative Success Rate
####################################################

running = []

count = 0

for i, s in enumerate(success, start=1):

    count += s

    running.append(100 * count / i)

plt.figure(figsize=(8,4))

plt.plot(running)

plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")

plt.title("Cumulative Success Rate")

plt.grid(True)

plt.tight_layout()

plt.savefig(PLOT_DIR / "cumulative_success.png")



####################################################
# Collision Rate (Rolling)
####################################################

window = 20

collision = [e["collision"] for e in episodes]

rolling_collision = []

for i in range(len(collision)):

    start = max(0, i-window+1)

    rolling_collision.append(
        100*np.mean(collision[start:i+1])
    )

plt.figure(figsize=(8,4))

plt.plot(rolling_collision)

plt.xlabel("Episode")

plt.ylabel("Collision Rate (%)")

plt.title("Rolling Collision Rate")

plt.grid(True)

plt.tight_layout()

plt.savefig(PLOT_DIR / "rolling_collision.png")


###################################################
# Boxplot of rewards
####################################################


plt.figure(figsize=(5,5))

plt.boxplot(rewards)

plt.ylabel("Reward")

plt.title("Reward Distribution")

plt.tight_layout()

plt.savefig(PLOT_DIR / "reward_boxplot.png")



####################################################
# Left Right Success
####################################################

left = [
    e for e in episodes
    if e["target"]=="LEFT"
]

right = [
    e for e in episodes
    if e["target"]=="RIGHT"
]

left_rate = (
    100*np.mean(
        [e["success"] for e in left]
    )
)

right_rate = (
    100*np.mean(
        [e["success"] for e in right]
    )
)

plt.figure(figsize=(6,4))

plt.bar(
    ["LEFT","RIGHT"],
    [
        left_rate,
        right_rate,
    ],
)

plt.ylabel("Success %")

plt.title("Target Direction Success")

plt.tight_layout()

plt.savefig("direction_success.png")

print("Plots saved.")

####################################################
# Direction Success Rate
####################################################


plt.figure(figsize=(6,4))

plt.bar(
    ["LEFT Episodes", "RIGHT Episodes"],
    [len(left), len(right)]
)

plt.ylabel("Count")

plt.title("Evaluation Episode Distribution")

plt.tight_layout()

plt.savefig(PLOT_DIR / "direction_counts.png")