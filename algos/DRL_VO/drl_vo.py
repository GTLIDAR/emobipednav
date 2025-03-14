import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

from algos.DRL_VO.utils import preprocess_obs, soft_update_params
from algos.DRL_VO.encoder import make_encoder


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

################################## PPO Policy ##################################

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.obses = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.obses[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_shape[0]), nn.Tanh()
        )

        self.apply(weight_init)

    def forward(self, obs, robot_goal_state, detach_encoder=True):
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        return self.trunk(obs)

class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, robot_goal_state_dim, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters
        )

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim + robot_goal_state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.apply(weight_init)

    def forward(self, obs, robot_goal_state, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        obs = torch.cat((obs, robot_goal_state), dim=-1)
        return self.trunk(obs)
    
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, action_std_init, device, robot_goal_state_dim, hidden_dim, encoder_type,
                       encoder_feature_dim, num_layers, num_filters):
        super(ActorCritic, self).__init__()

        self.device = device
        self.action_dim = action_shape[0]
        self.action_var = torch.full((self.action_dim,), action_std_init * action_std_init).to(device)
                
        self.actor = Actor(obs_shape, robot_goal_state_dim, action_shape, hidden_dim, encoder_type,
                           encoder_feature_dim, num_layers, num_filters)
        
        self.critic = Critic(obs_shape, robot_goal_state_dim, hidden_dim, encoder_type,
                             encoder_feature_dim, num_layers, num_filters)

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def forward(self):
        raise NotImplementedError
    
    def act(self, obs, robot_goal_state, eval):
        action_mean = self.actor(obs, robot_goal_state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
      
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(obs, robot_goal_state)

        if eval:
            return action_mean.detach(), action_logprob.detach(), state_val.detach()
        else:
            return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, obs, robot_goal_state, action):
        action_mean = self.actor(obs, robot_goal_state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
      
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs, robot_goal_state)
        
        return action_logprobs, state_values, dist_entropy
    
class PPOAgent(object):
    """PPO+Encoder algorithm."""
    def __init__(
        self,
        obs_shape,
        robot_goal_state_dim, 
        action_shape,
        action_range, 
        device,
        args
    ):
        self.action_shape = action_shape
        self.device = device
        self.action_std = args.action_std
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.K_epochs = args.K_epochs
        
        self.buffer = RolloutBuffer()
        # action rescaling
        self.action_scale = torch.FloatTensor((action_range[1] - action_range[0]) / 2.).to(device)
        self.action_bias = torch.FloatTensor((action_range[1] + action_range[0]) / 2.).to(device)

        self.policy = ActorCritic(obs_shape, action_shape, self.action_std, device, robot_goal_state_dim, args.hidden_dim, 
                                  args.encoder_type, args.encoder_feature_dim, args.num_layers, args.num_filters).to(device)

        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': args.actor_lr},
                        {'params': self.policy.critic.parameters(), 'lr': args.critic_lr}
                    ])

        self.policy_old = ActorCritic(obs_shape, action_shape, self.action_std, device, robot_goal_state_dim, args.hidden_dim, 
                                      args.encoder_type, args.encoder_feature_dim, args.num_layers, args.num_filters).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
        self.set_action_std(self.action_std)

    def select_action(self, obs, robot_goal_state, eval=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            robot_goal_state = torch.FloatTensor(robot_goal_state).to(self.device)
            robot_goal_state = robot_goal_state.unsqueeze(0)
            action, action_logprob, state_val = self.policy_old.act(obs, robot_goal_state, eval)

        if not eval:
            self.buffer.states.append(robot_goal_state)
            self.buffer.obses.append(obs)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

        action = action.clamp(-1.0, 1.0)
        action = action * self.action_scale + self.action_bias
        return action.detach().cpu().numpy().flatten()
    
    def update(self, writer, step):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_obses = torch.squeeze(torch.stack(self.buffer.obses, dim=0)).detach().to(self.device)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_obses, old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            writer.add_scalar('loss', loss.mean(), step)
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, filename):
        torch.save(self.policy_old.state_dict(), filename + '_ppo')

    def load(self, filename):
        self.policy_old.load_state_dict(torch.load(filename + '_ppo', map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(filename + '_ppo', map_location=lambda storage, loc: storage))
        
            