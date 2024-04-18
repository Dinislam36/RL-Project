from torch import nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Define convolutional layers for image processing (adjust based on your image size)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Define fully-connected layers for hidden representation
        self.fc = nn.Sequential(
            nn.Linear(
                64 * (state_dim[0] // 16) * (state_dim[1] // 16), 128
            ),  # Input based on conv output size
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Output layer for action probabilities (linear + tanh for probabilities)
        self.output = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)  # Flatten conv output for FC layers
        x = self.fc(x)
        # Use softmax for multiple discrete actions or tanh for continuous actions
        return self.softmax(self.output(x))


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # Define convolutional layers for image processing (similar to Actor)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Define fully-connected layers for hidden representation (similar to Actor)
        self.fc = nn.Sequential(
            nn.Linear(64 * (state_dim[0] // 16) * (state_dim[1] // 16), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Output layer for Q-value estimation (linear)
        self.output = nn.Linear(64, 1)

    def forward(self, state):
        x = self.conv(state)
        x = x.view(x.size(0), -1)  # Flatten conv output for FC layers
        return self.output(self.fc(x))
