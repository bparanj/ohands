# Hugging Face Tokenizer Demo

This project demonstrates how to work with tokens using the Hugging Face transformers library.

## Setup Instructions

### 1. Create and Activate Virtual Environment

#### For Windows:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```

#### For macOS/Linux:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Required Packages

Once your virtual environment is activated, install the required packages:

```bash
pip install transformers torch
```

Note: If you're using a CPU-only machine, you can install a lighter version of PyTorch:
```bash
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the Program

```bash
python token_demo.py
```

## Program Features

The program demonstrates various token operations:
1. Basic tokenization of text
2. Converting tokens to IDs and back
3. Batch tokenization with padding
4. Special tokens handling
5. Vocabulary access

## Requirements

- Python 3.8 or higher
- transformers
- torch (PyTorch)

## Expected Output

The program will show:
- How text is tokenized
- Token to ID mappings
- Batch processing examples
- Special tokens information
- Vocabulary examples

## Troubleshooting

1. If you get an error about CUDA not being available, don't worry - the program will run on CPU.

2. First time running might take longer as it downloads the BERT model.

3. If you get an import error:
   - Make sure your virtual environment is activated
   - Verify that all packages are installed: `pip list`

4. If you get a memory error:
   - Try running on a machine with more RAM
   - Or reduce the batch size in the code

## Additional Notes

- The program uses the 'bert-base-uncased' model by default
- Downloaded model files are cached locally
- Internet connection required for first run to download the model