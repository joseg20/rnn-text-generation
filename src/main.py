import argparse
import torch
from rnn_model import RNN
from train import train
from generate import generate_text
import json
# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/corpus.txt', help='Archivo de texto para entrenamiento')
parser.add_argument('--model_path', type=str, default='../models/trained_rnn_model.pth', help='Ruta al modelo entrenado')
parser.add_argument('--epochs', type=int, default=10, help='Número de épocas de entrenamiento')
parser.add_argument('--generate', action='store_true', help='Generar texto en lugar de entrenar un nuevo modelo')
parser.add_argument('--seed_text', type=str, default="Once upon a time", help='Texto inicial para generar el texto')
args = parser.parse_args()

if not args.generate:
    # Read and process data
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create a unique character set
    unique_chars = sorted(set(text))

    # Create the dictionaries char_to_idx and idx_to_char
    char_to_idx = {c: i for i, c in enumerate(unique_chars)}
    idx_to_char = {i: c for i, c in enumerate(unique_chars)}

    # Save the dictionaries as JSON files
    with open('../models/char_to_idx.json', 'w') as fp:
        json.dump(char_to_idx, fp)
    with open('../models/idx_to_char.json', 'w') as fp:
        json.dump(idx_to_char, fp)


    # Define vocab_size, hidden_size, num_layers and dropout
    vocab_size = len(unique_chars)
    hidden_size = 128
    num_layers = 2
    dropout = 0.5

    # Prepare the data: tokenize, create vocabulary and mappings, etc.
    sequence_length = 50

    # Create input sequences and target tags
    input_sequences = []
    target_sequences = []

    # Scroll through text and create character sequences and target tags
    for i in range(0, len(text) - sequence_length):
        input_seq = text[i:i + sequence_length]
        target_seq = text[i + 1:i + sequence_length + 1]
        input_sequences.append(torch.tensor([char_to_idx[c] for c in input_seq], dtype=torch.long))
        target_sequences.append(torch.tensor([char_to_idx[c] for c in target_seq], dtype=torch.long))

    # Pack input sequences and target tags into tensor pairs.
    data = list(zip(input_sequences, target_sequences))

    # Create an instance of the RNN model
    model = RNN(vocab_size, hidden_size, num_layers, dropout)

    # Training the model
    train(model, data, args.epochs)

    # Save the trained model
    torch.save(model.state_dict(), '../models/trained_rnn_model.pth')

else:
    # Load the dictionaries from JSON files
    with open('../models/char_to_idx.json', 'r') as fp:
        char_to_idx = json.load(fp)
    with open('../models/idx_to_char.json', 'r') as fp:
        idx_to_char = {int(k): v for k, v in json.load(fp).items()}



    # Define vocab_size, hidden_size, num_layers and dropout
    vocab_size = len(char_to_idx)
    hidden_size = 128
    num_layers = 2
    dropout = 0.5

    # Load the trained model
    loaded_model = RNN(vocab_size, hidden_size, num_layers, dropout)
    loaded_model.load_state_dict(torch.load(args.model_path))

    # Generate text with the trained model
    seed_text = args.seed_text
    generated_text = generate_text(loaded_model, seed_text, char_to_idx, idx_to_char)
    print("Generated text:", generated_text)
