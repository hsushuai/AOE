<h1 align="center">
<img src="misc/game.svg" width="100" alt="rho-logo" />
<br>
Strategy-Augmented Planning for Large Language Models via Opponent Exploitation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2505.08459">
    <img src="https://img.shields.io/badge/arXiv-2505.08459-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://github.com/hsushuai/SAP/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/hsushuai/SAP.svg" alt="License">
  </a>
  <a href="https://github.com/hsushuai/SAP">
    <img src="https://img.shields.io/github/stars/hsushuai/SAP?style=social" alt="GitHub stars">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python version">
  <img src="https://img.shields.io/badge/Env-uv%20%7C%20venv-green" alt="Environment">
</p>


<p align="center">
<img src=misc/framework.svg width=700/>
</p>

In this work,  we introduce a two-stage **Strategy-Augmented Planning** (**SAP**) framework that significantly enhances the opponent exploitation capabilities of LLM-based agents by utilizing a critical component, the Strategy Evaluation Network (SEN). Specifically, in the offline stage, we construct an explicit strategy space and subsequently collect strategy-outcome pair data for training the SEN network. During the online phase, SAP dynamically recognizes the opponent's strategies and greedily exploits them by searching best response strategy on the well-trained SEN, finally translating strategy to a course of actions by carefully designed prompts.

## 📦 Setup

We recommend using [uv](https://docs.astral.sh/uv) to manage the environment. If you don’t have `uv` installed, you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then set up the project by cloning the repository and syncing dependencies:

```bash
git clone https://github.com/hsushuai/SAP.git
cd SAP
uv sync
```

This will automatically create a virtual environment in .venv and install dependencies from uv lock files.

To activate the environment:

```bash
source .venv/bin/activate
```

## ⚡ Quick Start

To run the MicroRTS environment, ensure an **X server** is running:

* On Windows: we recommend [VcXsrv](https://sourceforge.net/projects/vcxsrv/)
* On macOS: install [XQuartz](https://www.xquartz.org/)
* On Linux: usually built-in

If you're using SSH on a remote server, enable X11 forwarding with:

```bash
ssh -X user@remote-server
```

Then, set the `DISPLAY` environment variable:

```bash
export DISPLAY=<YOUR_HOST_IP>:<PORT>
```

> Replace `<YOUR_HOST_IP>` and `<PORT>` with your actual X server settings.
> For example: `export DISPLAY=localhost:0`

To verify the environment is working correctly, run the following test script:

```bash
python skill_rts/hello_world.py
```

You should see a simple MicroRTS environment window launch successfully.

## 🔬 Experiments

To run the experiments, you can use the provided scripts. For example, to evaluate the performance of the SAP agent, run:

```bash
python sap/experiments/eval_sap.py
```

More scripts are available in the `sap/experiments` directory for different tasks and configurations.

## 📂 Structure

There are three main directories in this repository: `microrts`, `sap` and `skill_rts`.

- `microrts`: Contains the MicroRTS environment implementation and related utilities.
- `sap`: Contains the Strategy-Augmented Planning framework implementation, including the Strategy Evaluation Network (SEN) and related components.
- `skill_rts`: Contains the high-level skills and utilities for interacting with the MicroRTS environment.

This project builds on the foundations of [MicroRTS-Py](https://github.com/Farama-Foundation/MicroRTS-Py) and [PLAP](https://github.com/AI-Research-TeamX/PLAP). We gratefully acknowledge their contributions.

## 📄 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{xu2025sap,
      title={Strategy-Augmented Planning for Large Language Models via Opponent Exploitation}, 
      author={Shuai Xu and Sijia Cui and Yanna Wang and Bo Xu and Qi Wang},
      year={2025},
      eprint={2505.08459},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.08459}, 
}
```