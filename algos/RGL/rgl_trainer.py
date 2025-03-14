import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class RGLTrainer(object):
    def __init__(self, gamma, time_step, v_pref,
                 value_estimator, state_predictor, memory, device, policy, writer, batch_size, optimizer_str, human_num,
                 reduce_sp_update_frequency, freeze_state_predictor, detach_state_predictor):
        """
        Train the trainable model of a policy
        """
        self.value_estimator = value_estimator
        self.state_predictor = state_predictor
        self.device = device
        self.writer = writer
        self.target_policy = policy
        self.target_model = None
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer_str = optimizer_str
        self.reduce_sp_update_frequency = reduce_sp_update_frequency
        self.state_predictor_update_interval = human_num
        self.freeze_state_predictor = freeze_state_predictor
        self.detach_state_predictor = detach_state_predictor
        self.v_optimizer = None
        self.s_optimizer = None

        # for value update
        self.gamma = gamma
        self.time_step = time_step
        self.v_pref = v_pref

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def set_learning_rate(self, learning_rate):
        if self.optimizer_str == 'Adam':
            self.v_optimizer = optim.Adam(self.value_estimator.parameters(), lr=learning_rate)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.Adam(self.state_predictor.parameters(), lr=learning_rate)
        elif self.optimizer_str == 'SGD':
            self.v_optimizer = optim.SGD(self.value_estimator.parameters(), lr=learning_rate, momentum=0.9)
            if self.state_predictor.trainable:
                self.s_optimizer = optim.SGD(self.state_predictor.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

    def optimize_epoch(self, num_epochs):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)

        for epoch in range(num_epochs):
            epoch_v_loss = 0
            epoch_s_loss = 0

            update_counter = 0
            for data in self.data_loader:
                robot_states, human_states, values, _, _, next_human_states = data

                # optimize value estimator
                self.v_optimizer.zero_grad()
                outputs = self.value_estimator((robot_states, human_states))
                values = values.to(self.device)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.v_optimizer.step()
                epoch_v_loss = epoch_v_loss + loss.data.item()

                # optimize state predictor
                if self.state_predictor.trainable:
                    update_state_predictor = True
                    if update_counter % self.state_predictor_update_interval != 0:
                        update_state_predictor = False

                    if update_state_predictor:
                        self.s_optimizer.zero_grad()
                        _, next_human_states_est = self.state_predictor((robot_states, human_states), None, None)
                        loss = self.criterion(next_human_states_est, next_human_states)
                        loss.backward()
                        self.s_optimizer.step()
                        epoch_s_loss =  epoch_s_loss + loss.data.item()
                    update_counter += 1

            self.writer.add_scalar('IL/epoch_v_loss', epoch_v_loss / len(self.memory), epoch)
            self.writer.add_scalar('IL/epoch_s_loss', epoch_s_loss / len(self.memory), epoch)


        return

    def optimize_batch(self, episode):
        if self.v_optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        v_losses = 0
        s_losses = 0
        batch_count = 0
        for data in self.data_loader:
            robot_states, human_states, _, rewards, next_robot_states, next_human_states = data

            # optimize value estimator
            self.v_optimizer.zero_grad()
            outputs = self.value_estimator((robot_states, human_states))

            gamma_bar = pow(self.gamma, self.time_step * self.v_pref)
            target_values = rewards + gamma_bar * self.target_model((next_robot_states, next_human_states))

            # values = values.to(self.device)
            loss = self.criterion(outputs, target_values)
            loss.backward()
            self.v_optimizer.step()
            v_losses = v_losses + loss.data.item()

            # optimize state predictor
            if self.state_predictor.trainable:
                update_state_predictor = True
                if self.freeze_state_predictor:
                    update_state_predictor = False
                elif self.reduce_sp_update_frequency and batch_count % self.state_predictor_update_interval == 0:
                    update_state_predictor = False

                if update_state_predictor:
                    self.s_optimizer.zero_grad()
                    _, next_human_states_est = self.state_predictor((robot_states, human_states), None, None,
                                                                    detach=self.detach_state_predictor)
                    loss = self.criterion(next_human_states_est, next_human_states)
                    loss.backward()
                    self.s_optimizer.step()
                    s_losses = s_losses + loss.data.item()

            batch_count += 1


        average_v_loss = v_losses / self.batch_size
        average_s_loss = s_losses / self.batch_size
        self.writer.add_scalar('RL/average_v_loss', average_v_loss, episode)
        self.writer.add_scalar('RL/average_s_loss', average_s_loss, episode)

        return average_v_loss, average_s_loss