import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.device = torch.device("cuda")
        self.hidden = nn.Linear(input_size, hidden_size).to(self.device)  # Hidden layer
        self.output = nn.Linear(hidden_size, output_size).to(self.device)  # Output layer
        self.activation = nn.Tanh().to(self.device)  # Activation function

    def forward(self, x):
        x = self.activation(self.hidden(x))  # Activation function for hidden layer
        x = self.output(x)  # Output layer
        return x

    def copy(self):
        """Returns a deep copy of this neural network."""
        new_network = NeuralNetwork(self.hidden.in_features,
                                    self.hidden.out_features,
                                    self.output.out_features)
        new_network.load_state_dict(self.state_dict())
        return new_network
