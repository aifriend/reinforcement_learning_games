"""Smoke tests: each CartPole algorithm imports, builds its agent and takes one env step.

Pong, LunarLander and BipedalWalker scripts are not covered: they need Atari
ROMs, Box2D, or long-running multiprocess training.
"""
import numpy as np
import pytest

CARTPOLE_SCRIPTS = [
    pytest.param("01_CartPole-reinforcement-learning", "Cartpole_DQN.py", False, id="dqn"),
    pytest.param("02_CartPole-reinforcement-learning_DDQN", "Cartpole_DDQN_TF2.py", True, id="ddqn"),
    pytest.param("03_CartPole-reinforcement-learning_Dueling_DDQN", "Cartpole_Double_DDQN_TF2.py", True, id="dueling-ddqn"),
    pytest.param("04_CartPole-reinforcement-learning_e_greedy_D3QN", "Cartpole_e_greedy_D3QN_TF2.py", True, id="e-greedy-d3qn"),
]


@pytest.mark.parametrize("folder,filename,takes_env_name", CARTPOLE_SCRIPTS)
def test_cartpole_agent_one_step(load_script, folder, filename, takes_env_name):
    module = load_script(folder, filename)
    agent = module.DQNAgent("CartPole-v1") if takes_env_name else module.DQNAgent()

    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])
    if filename.startswith("Cartpole_e_greedy"):
        action, _explore_probability = agent.act(state, 1)  # also returns the exploration rate
    else:
        action = agent.act(state)
    assert action in range(agent.action_size)

    next_state, reward, done, _ = agent.env.step(action)
    assert np.shape(next_state) == (agent.state_size,)
    assert isinstance(done, (bool, np.bool_))

    agent.remember(state, action, reward, np.reshape(next_state, [1, agent.state_size]), done)
    assert len(agent.memory) == 1


def test_cartpole_per_d3qn_one_step(load_script):
    module = load_script("05_CartPole-reinforcement-learning_PER_D3QN", "Cartpole_PER_D3QN_TF2.py")
    agent = module.DQNAgent("CartPole-v1")

    state = agent.env.reset()
    state = np.reshape(state, [1, agent.state_size])
    action, _explore_probability = agent.act(state, 1)
    assert action in range(agent.action_size)

    next_state, reward, done, _ = agent.env.step(action)
    assert np.shape(next_state) == (agent.state_size,)

    agent.remember(state, action, reward, np.reshape(next_state, [1, agent.state_size]), done)
