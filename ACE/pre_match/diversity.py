import numpy as np
import pandas as pd
import requests
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json


def load_strategies():
    strategies = []
    for i in range(1, 20):
        with open(f"ACE/data/opponent_strategy_{i}.json") as f:
            strategies.append(json.load(f)["strategy"])
    return strategies


def parse_strategy_features(strategy):
    feats = []


def main():
    pass


if __name__ == "__main__":
    main()
