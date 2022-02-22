import torch
import torch.nn as nn
import torch.nn.functional as F

class dueling_DQN(nn.Module):
  """ Actor Model. """

  def __init__(self, state_size, action_size, seed, h1_size=64, h2_size=64):
    """ Initialize parameters and build model
    Params
    ======
      state_size (int): Dimension of each state
      action_size (int): Dimension of each action
      seed (int): Random seed
    """

    super(dueling_DQN, self).__init__()
    self.action_size = action_size
    h3_size = h2_size//2
    self.seed = torch.manual_seed(seed)
    # Define Network Model
    self.h1 = nn.Linear(state_size, h1_size)
    self.h2 = nn.Linear(h1_size, h2_size)
    
    # Separate into 2 streams
    # Value stream
    self.h3_V = nn.Linear(h2_size, h3_size)
    self.h4_V = nn.Linear(h3_size, 1)
    # Advantage stream
    self.h3_A = nn.Linear(h2_size, h3_size)
    self.h4_A = nn.Linear(h3_size, action_size)
    

  def forward(self, state):
    # State -> action values, network 
    x = F.relu(self.h1(state))
    x = F.relu(self.h2(x))
    
    # Value stream
    V = F.relu(self.h3_V(x))
    V = self.h4_V(V)
    # Advantage stream
    A = F.relu(self.h3_A(x))
    A = self.h4_A(A)
    
    Q = V + A - A.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)
    return Q