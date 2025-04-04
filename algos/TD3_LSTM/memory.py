import numpy as np
import torch
import random

class eval_policy_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ReplayBuffer(object):
    def __init__(
        self, lidar_state_dim, position_state_dim, action_dim, hidden_size, max_size, device
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.lidar_state = np.zeros((self.max_size, lidar_state_dim), dtype=np.float32)
        self.position_robot_state = np.zeros((self.max_size, position_state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_lidar_state = np.zeros((self.max_size, lidar_state_dim), dtype=np.float32)
        self.next_position_state = np.zeros((self.max_size, position_state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)


        self.h = np.zeros((self.max_size, hidden_size), dtype=np.float32)
        self.nh = np.zeros((self.max_size, hidden_size), dtype=np.float32)

        self.c = np.zeros((self.max_size, hidden_size), dtype=np.float32)
        self.nc = np.zeros((self.max_size, hidden_size), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self, lidar_state, position_robot_state, action, next_lidar_state, next_position_state, 
              reward, done, hiddens, next_hiddens
    ):
        self.lidar_state[self.ptr] = lidar_state
        self.position_robot_state[self.ptr] = position_robot_state
        self.action[self.ptr] = action
        self.next_lidar_state[self.ptr] = next_lidar_state
        self.next_position_state[self.ptr] = next_position_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

      
        h, c = hiddens
        nh, nc = next_hiddens

        # Detach the hidden state so that BPTT only goes through 1 timestep
        self.h[self.ptr] = h.detach().cpu().numpy().flatten()
        self.c[self.ptr] = c.detach().cpu().numpy().flatten()
        self.nh[self.ptr] = nh.detach().cpu().numpy().flatten()
        self.nc[self.ptr] = nc.detach().cpu().numpy().flatten()

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=100):
        # TODO: Clean this up. There's probably a cleaner way to seperate
        # on-policy and off-policy sampling. Clean up extra-dimension indexing
        # also
        ind = np.random.randint(0, self.size, size=int(batch_size))


        h = torch.tensor(self.h[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float32).to(self.device)
        c = torch.tensor(self.c[ind][None, ...],
                         requires_grad=True,
                         dtype=torch.float32).to(self.device)
        nh = torch.tensor(self.nh[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float32).to(self.device)
        nc = torch.tensor(self.nc[ind][None, ...],
                          requires_grad=True,
                          dtype=torch.float32).to(self.device)

        # TODO: Return hidden states or not, or only return the
        # first hidden state (although it's already been detached,
        # so returning nothing might be better)
        hidden = (h, c)
        next_hidden = (nh, nc)

        l_s = torch.FloatTensor(
            self.lidar_state[ind][:, None, :]).to(self.device)
        p_s = torch.FloatTensor(
            self.position_robot_state[ind][:, None, :]).to(self.device)
        a = torch.FloatTensor(
            self.action[ind][:, None, :]).to(self.device)
        n_l_s = torch.FloatTensor(
            self.next_lidar_state[ind][:, None, :]).to(self.device)
        n_p_s = torch.FloatTensor(
            self.next_position_state[ind][:, None, :]).to(self.device)
        r = torch.FloatTensor(
            self.reward[ind][:, None, :]).to(self.device)
        d = torch.FloatTensor(
            self.not_done[ind][:, None, :]).to(self.device)

        return l_s, p_s, a, n_l_s, n_p_s, r, d, hidden, next_hidden

