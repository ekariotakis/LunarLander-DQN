# LunarLander-DQN
This project is part of the course **Autonomous Agents** at School of Electrical and Computer Engineering, Technical University of Crete.

The purpose of this project is the understanding of Q-Learning, a Reinforcement Learning Algorithm, 
and the implementation of a Deep Q-Learning agent for solving a control task.

The task we sought to solve is “LunarLander-v2”, by OpenAI Gym.
The goal of this task is to land a small spacecraft in between two of two flags.

# Code
There are two main folders, `./dqn` and `./dueling_dqn` which implement a simple DQN agent and a dueling-DQN agent, accordingly.

(you can find more information about them in the `Report.pdf` file, in the section "Implementation")

# Setup
In order to run the project, you need to have an environment with installed
- `Python 3.9.10`
- `PyTorch 1.10.2`
- `OpenAI Gym`

***Note:*** In order to run the `LunarLander-v2` you need to install `box2d-py`, which does not come along with the installation of `OpenAI Gym`.

# Tensorboard Logs
The tensorboard logs are stored in folder `./logs` in the main project directory.
In order to see those logs, you have to run tensorboard. To do that, open an command prompt at the directory of the project, where `./logs` is located.
Then run the following command:
```
tensorboard --logdir=logs
```
This will open an http port in locahost that will allow you to view the logs. The address that you generally need to visit from any browser is `http://localhost:6006` to see the detailed logs.

The name of the log that is being created at each run of the algortihm contains the date that the model got trained.

# Saved Models and Visualization
The models that are being trained are saved in the folder `./models`, in the `./dqn` or `./dueling_dqn` directory, depending on which architecture you are using for training. We store a model for every 100 episodes at a subfolder of `./models` with a name based again on the date that the model got trained.
In order to visualize the behaviour of each agent you just need to chane the input of the `load` function in the **Visualization** cell to the one of your choice and run it.
