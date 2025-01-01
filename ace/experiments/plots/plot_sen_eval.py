import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_mat():
    confusion_mat = [
        [0.84, 0.16],
        [0.26, 0.74]
    ]
    plt.rcParams.update({"font.size": 20})

    labels = ["0", "1"]
    tick_labels = ["0", "1"]
    sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=tick_labels, annot_kws={"size": 20})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("ace/experiments/plots/confusion_matrix.pdf")


def plot_win_loss():
    # Data
    runs_dir = "runs/eval_sen"
    win_loss = []
    for filename in os.listdir(runs_dir):
        with open(f"{runs_dir}/{filename}/metric.json") as f:
            metric = json.load(f)
        win_loss.append(metric["win_loss"][0])
    num_win, num_draw, num_loss = 0, 0, 0
    for wl in win_loss:
        if wl > 0:
            num_win += 1
        elif wl == 0:
            num_draw += 1
        else:
            num_loss += 1
    data = [num_win / len(win_loss), num_draw / len(win_loss), num_loss / len(win_loss)]
    data = [d * 100 for d in data]
    labels = ["Win", "Draw", "Loss"]
    colors = ["#d86c50"]  # "#d86c50", "#0ac9bf", "#a39aef"
    bar_width = 0.5
    plt.rcParams.update({"font.size": 20})

    # Plot
    plt.clf()
    plt.barh(labels[::-1], data[::-1], color=colors[::-1], hatch="/", edgecolor="black", height=bar_width)
    plt.xlabel("Rate (%)")

    # Save plot
    plt.tight_layout()
    plt.savefig("ace/experiments/plots/eval_sen.pdf")


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    plot_confusion_mat()
    plot_win_loss()
