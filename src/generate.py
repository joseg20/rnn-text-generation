import torch
import torch.nn.functional as F
import numpy as np

def generate_text(model, seed_text,char_to_idx, idx_to_char, length=100, temperature=1.0):
    model.eval()

    # Convert initial text to a sequence of tokens
    input_sequence = [char_to_idx[c] for c in seed_text]

    # Create a tensor from the input sequence
    input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0)

    # Initialize hidden state
    hidden = None

    # Generate text
    generated_text = seed_text
    for _ in range(length):
        # Pass the input through the model
        output, hidden = model(input_tensor, hidden)

        # Obtain the probability distribution of the last character
        probs = F.softmax(output[:, -1] / temperature, dim=-1).squeeze()

        # Selecting a characteristic according to the probability distribution
        next_char_idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char[next_char_idx]

        # Add the generated character to the text
        generated_text += next_char

        # Update the input sequence for the next iteration
        input_tensor = torch.tensor([next_char_idx], dtype=torch.long).unsqueeze(0)

    return generated_text
