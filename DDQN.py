class DQNNetwork(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(64 * 29 * 29, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DDQNAgent:
    def __init__(self, state_space, action_space, gamma=0.99, lr=1e-3, batch_size=64):
        self.online_net = DQNNetwork(state_space, action_space)
        self.target_net = DQNNetwork(state_space, action_space)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = []
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.action_space = action_space

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > 10000:
            self.replay_buffer.pop(0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states).permute(0, 3, 1, 2).float()
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).float()
        next_states = torch.tensor(next_states).permute(0, 3, 1, 2).float()
        dones = torch.tensor(dones).float()

        q_values = self.online_net(states).gather(1, actions).squeeze(1)
        next_q_values_online = self.online_net(next_states).max(1)[1]
        next_q_values_target = self.target_net(next_states).gather(1, next_q_values_online.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + (self.gamma * next_q_values_target * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def select_action(self, state, epsilon=0.05):
        if random.random() < epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            state = torch.tensor(state).permute(2, 0, 1).unsqueeze(0).float()
            q_values = self.online_net(state)
            return q_values.max(1)[1].item()

# Initialize the agent
