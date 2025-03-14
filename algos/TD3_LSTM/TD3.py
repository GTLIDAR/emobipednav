import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


print(torch.cuda.is_available(), torch.backends.cudnn.enabled)


class Actor(nn.Module):
    def __init__(
        self, lidar_state_dim, robot_goal_state_dim, lidar_feature_dim, action_dim, hidden_dim, action_range
    ):
        super(Actor, self).__init__()
      

        self.lidar_compress_net = nn.Sequential(
            nn.Linear(lidar_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, lidar_feature_dim, nn.ReLU())
        )


        self.l1 = nn.LSTM(lidar_feature_dim + robot_goal_state_dim, hidden_dim, batch_first=True)
     

        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.action_scale = torch.FloatTensor(
            (action_range[1] - action_range[0]) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_range[1] + action_range[0]) / 2.)

    def forward(self, lidar_state, position_robot_state, hidden):
        with torch.no_grad():
            lidar_feature = self.lidar_compress_net(lidar_state)
        state = torch.cat((lidar_feature, position_robot_state), dim=-1)

        self.l1.flatten_parameters()
        a, h = self.l1(state, hidden)


        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        a = a * self.action_scale + self.action_bias
        return a, h

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

class Critic(nn.Module):
    def __init__(
        self, lidar_state_dim, robot_goal_state_dim, lidar_feature_dim, action_dim, hidden_dim
    ):
        super(Critic, self).__init__()
      
        self.lidar_compress_net = nn.Sequential(
            nn.Linear(lidar_state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, lidar_feature_dim, nn.ReLU())
        )


        self.l1 = nn.LSTM(
            lidar_feature_dim + robot_goal_state_dim + action_dim, hidden_dim, batch_first=True)
        self.l4 = nn.LSTM(
            lidar_feature_dim + robot_goal_state_dim + action_dim, hidden_dim, batch_first=True)

      

        # Q1 architecture
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, lidar_state, position_robot_state, action, hidden1, hidden2):
        lidar_feature = self.lidar_compress_net(lidar_state)
        state = torch.cat((lidar_feature, position_robot_state), -1)
        sa = torch.cat([state, action], -1)

        self.l1.flatten_parameters()
        self.l4.flatten_parameters()
        q1, hidden1 = self.l1(sa, hidden1)
        q2, hidden2 = self.l4(sa, hidden2)
      

        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, lidar_state, position_robot_state, action, hidden1):
        lidar_feature = self.lidar_compress_net(lidar_state)
        state = torch.cat((lidar_feature, position_robot_state), -1)
        sa = torch.cat([state, action], -1)
        self.l1.flatten_parameters()
        q1, hidden1 = self.l1(sa, hidden1)


        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):
    def __init__(
            self,
            lidar_state_dim,
            robot_goal_state_dim,
            lidar_feature_dim,
            action_dim,
            action_range,
            hidden_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            lr=3e-4,
            device='cpu'
    ):
        self.device = torch.device(device)

        self.actor = Actor(
            lidar_state_dim, robot_goal_state_dim, lidar_feature_dim, action_dim, hidden_dim, action_range
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=lr)

        self.critic = Critic(
            lidar_state_dim, robot_goal_state_dim, lidar_feature_dim, action_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.action_range = torch.FloatTensor(action_range).to(self.device)
        self.discount = discount
        self.tau = tau
        self.policy_noise = torch.FloatTensor(policy_noise).to(self.device)
        self.noise_clip = torch.FloatTensor(noise_clip).to(self.device)
        self.policy_freq = policy_freq

        self.total_it = 0

    def get_initial_states(self):
        h_0, c_0 = None, None
       
        h_0 = torch.zeros((
            self.actor.l1.num_layers,
            1,
            self.actor.l1.hidden_size),
            dtype=torch.float32)
        h_0 = h_0.to(device=self.device)

        c_0 = torch.zeros((
            self.actor.l1.num_layers,
            1,
            self.actor.l1.hidden_size),
            dtype=torch.float32)
        c_0 = c_0.to(device=self.device)
        return (h_0, c_0)

    def select_action(self, lidar_state, position_robot_state, hidden, test=True):

        lidar_state = torch.FloatTensor(
            lidar_state.reshape(1, -1)).to(self.device)[:, None, :]
        position_robot_state = torch.FloatTensor(
            position_robot_state.reshape(1, -1)).to(self.device)[:, None, :]

        action, hidden = self.actor(lidar_state, position_robot_state, hidden)
        return action.cpu().numpy().flatten(), hidden

    def train(self, replay_buffer, writer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        lidar_state, position_robot_state, action, next_lidar_state, next_position_state, reward, not_done, hidden, next_hidden = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_lidar_state, next_position_state, next_hidden)[0] + noise
            ).clamp(self.action_range[0], self.action_range[1])

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(
                next_lidar_state, next_position_state, next_action, next_hidden, next_hidden)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(lidar_state, position_robot_state, action, hidden, hidden)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        writer.add_scalar('train/critic_loss', critic_loss, self.total_it)

        for param, target_param in zip(
                self.critic.lidar_compress_net.parameters(), self.actor.lidar_compress_net.parameters()
            ):
            target_param.data.copy_(param.data)

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(
                lidar_state.detach(), position_robot_state.detach(), 
                self.actor(lidar_state.detach(), position_robot_state.detach(), hidden)[0], hidden).mean()
            writer.add_scalar('train/actor_loss', actor_loss, self.total_it)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")
        torch.save(self.actor_target.state_dict(), filename + "_actor_target")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor_target"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        
    def set_parameters(self, source):
        self.actor.load_state_dict(source.actor.state_dict())
        self.critic.load_state_dict(source.critic.state_dict())
        self.actor_target.load_state_dict(source.actor_target.state_dict())
        self.critic_target.load_state_dict(source.critic_target.state_dict())

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
