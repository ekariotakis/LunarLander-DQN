import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
  """ Actor Model. """

  def __init__(self, state_size, action_size, seed, h1_size=64, h2_size=64):
    """ Initialize parameters and build model
    Params
    ======
      state_size (int): Dimension of each state
      action_size (int): Dimension of each action
      seed (int): Random seed
    """

    super(DQN, self).__init__()
    self.seed = torch.manual_seed(seed)
    # Define Network Model
    self.h1 = nn.Linear(state_size, h1_size)
    self.h2 = nn.Linear(h1_size, h2_size)
    self.h3 = nn.Linear(h2_size, action_size)

  def forward(self, state):
    # State -> action values, network 
    x = F.relu(self.h1(state))
    x = F.relu(self.h2(x))
    return self.h3(x)