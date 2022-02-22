# LunarLander-DQN




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
The models that are being trained are saved in the folder `./models` in the main project directory. We store a model for every 100 episodes at a subfolder of `./models` with a name based again on the date that the model got trained.
In order to visualize the behaviour of each agent you just need to chane the input of the `load` function in the **Visualization** cell to the one of your choice and run it.
