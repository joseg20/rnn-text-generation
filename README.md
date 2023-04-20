
# rnn-text-generation

This project uses a Recurrent Neural Network (RNN) to generate text based on a training corpus. The model is trained on characters rather than words, allowing for greater flexibility in text generation.

## Run Locally

Clone the project

```bash
  git clone https://github.com/joseg20/rnn-text-generation
```

Go to the project directory

```bash
  cd rnn-text-generation
```

Install requirements

```bash
  pip install -r requirements.txt
```

Go to the source directory

```bash
  cd rnn-text-generation/src
```

 Prepare your training data in a text file (e.g., `data/corpus.txt`).

Train the model:
```bash
  python main.py --data data/corpus.txt --epochs 10
```

This will train the model on the specified text file and save the trained model in `models/trained_rnn_model.pth`.

Generate text using the trained model:
```bash
  python main.py --generate --seed_text "Once upon a time" --model_path models/trained_rnn_model.pth
```
This will generate text using the trained model and the seed text "Once upon a time".

## Usage

You can customize the training and generation process by adjusting the arguments in `main.py`. Some of the available arguments include:

- `--data`: Path to the text file for training.
- `--model_path`: Path to the trained model.
- `--epochs`: Number of training epochs.
- `--generate`: Generate text instead of training a new model.
- `--seed_text`: Seed text for generating text.

Refer to `main.py` for more options and details on how to adjust the model's hyperparameters.

## Contributing

Contributions are welcome. Please feel free to open an issue or a pull request if you find any bugs or have suggestions for improvements.
## Authors

- [@joseg20](https://github.com/joseg20)


## License

[MIT](https://choosealicense.com/licenses/mit/)

