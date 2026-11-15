# Reinforcement Learning Games:

This repository is adapted from [pythonlessons/Reinforcement_Learning](https://github.com/pythonlessons/Reinforcement_Learning).

PPO and PPO_CNN agents playing Pong-v0 game:<br>
![PPO agent](11_Pong-v0_PPO/gameplay.gif)
![PPO CNN agent](11_Pong-v0_PPO/gameplay_CNN.gif)

*2020-10-10 added LunarLander-v2_PPO Continuous code for Tensorflow 2.3.1*:
![LunarLander-v2_PPO](LunarLander-v2_PPO/gameplay.gif)

*2020-10-23 added BipedalWalker-v3_PPO code for Tensorflow 2.3.1*:
![BipedalWalker-v3_PPO_PPO](BipedalWalker-v3_PPO/gameplay.gif)

## Setup

Install the dependencies listed in `requirements.txt` into a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Each folder is self-contained and holds one agent (for example `01_CartPole-reinforcement-learning/Cartpole_DQN.py`).
The scripts take no command-line arguments. Run a script from inside its own folder, because model and image paths are relative:

```
cd 11_Pong-v0_PPO
python Pong-v0_PPO_TF2.py
```

What the script does is chosen at the bottom of the file, in the `if __name__ == "__main__":` block.
Uncomment the call you want, such as `agent.run()` to train or `agent.test(...)` to play with a saved model, and comment out the others.
Trained models (`.h5`) and learning curves (`.png`) are stored in the same folders.

1. [Deep Q Learning tutorial (DQN)](https://pylessons.com/CartPole-reinforcement-learning/)

2. [Double Deep Q Learning tutorial (DDQN)](https://pylessons.com/CartPole-DDQN/)

3. [Dueling Double Deep Q Learning tutorial (D3QN)](https://pylessons.com/CartPole-DDDQN/)

4. [Epsilon Greedy Dueling Double Deep Q Learning tutorial (D3QN)](https://pylessons.com/Epsilon-Greedy-DQN/)

5. [Prioritized Experience Replay (PER) D3QN tutorial](https://pylessons.com/CartPole-PER/)

6. [D3QN PER with Convolutional Neural Networks tutorial](https://pylessons.com/CartPole-PER-CNN/)

7. [A.I. learns to play Pong with DQN](https://pylessons.com/DQN-PONG/)

8. [Introduction to RL Policy Gradient (PG or REINFORCE)](https://pylessons.com/Beyond-DQN/)

9. [Introduction to RL Advanced Actor Critic algorythm (A2C)](https://pylessons.com/A2C-reinforcement-learning/)

10. [Introduction to RL Asynchronous Advanced Actor Critic algorythm (A3C)](https://pylessons.com/A3C-reinforcement-learning/)

11. [Introduction to RL Proximal Policy Optimization algorythm (PPO)](https://pylessons.com/PPO-reinforcement-learning/)

12. [Let’s code from scratch a discrete Reinforcement Learning rocket landing agent! (PPO)](https://pylessons.com/LunarLander-v2-PPO/)

13. [Continuous Proximal Policy Optimization Tutorial with OpenAI gym environment! (PPO)](https://pylessons.com/BipedalWalker-v3-PPO/)
<br><br>
PPO Pong-v0 Learning curve:
<img src="11_Pong-v0_PPO/Pong-v0_APPO_0.0001_RMSprop.png" data-canonical-src="11_Pong-v0_PPO/Pong-v0_APPO_0.0001_RMSprop.png" width="500" height="300" />
