import torch
import torch.optim as optim
import torch.nn as nn


def train(model, data, epochs):
    # Optimizer and loss function creation
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Initialization of hidden state
        hidden = None

        # Training loop
        for batch in data:
            # Input and  output preparation
            inputs, targets = batch

            # Input through the model
            output, hidden = model(inputs, hidden)
            
            # Detach hidden state from the computation graph
            hidden = hidden.detach()
            
            # Loss calculation
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))

            # Backpay and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss and other details
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')
