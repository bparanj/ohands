# Sequence and Summarization Demo

This program demonstrates text tokenization and summarization using the BART model from Hugging Face transformers.

## Features

1. Text Tokenization
   - Converts text into tokens
   - Shows token IDs and sequences
   - Demonstrates special tokens

2. Text Summarization
   - Uses BART model for summarization
   - Implements beam search
   - Controls summary length and quality

3. Token Analysis
   - Shows vocabulary information
   - Displays special tokens
   - Demonstrates attention masks

## Setup Instructions

### 1. Create Virtual Environment

#### Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

```bash
pip install transformers torch sentencepiece
```

For CPU-only installation:
```bash
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu sentencepiece
```

### 3. Run the Program

```bash
python sequence_summary.py
```

## Technical Details

The program uses:
- BART Large CNN model for summarization
- Beam search for better summary quality
- Attention mechanisms for sequence processing

## Expected Output

The program shows five main components:

1. **Tokenization Process**:
   - Shows the original text length
   - Number of tokens created
   - First 20 tokens (including special characters like Ġ which represents spaces)
   - Demonstrates how words are split into subwords

2. **Token IDs**:
   - Shows the shape of the input tensor (e.g., [1, 162] for one sequence of 162 tokens)
   - Displays the first few token IDs
   - Demonstrates how text is converted to numerical format

3. **Generated Summary**:
   - Shows the original input text
   - Displays the model-generated summary
   - Demonstrates the model's text generation capabilities
   - Shows how beam search affects the output

4. **Token Analysis**:
   - Lists all special tokens (e.g., <s>, </s>, <unk>, <pad>, <mask>)
   - Shows their corresponding IDs
   - Displays the total vocabulary size (50,265 tokens)
   - Explains the role of each special token

5. **Input Structure**:
   - Shows the attention mask shape
   - Demonstrates how the model knows which tokens to focus on
   - Matches the input_ids shape for proper processing

## Memory Requirements

- The BART base model is ~560MB
- Requires at least 2GB of RAM
- GPU is optional but recommended for faster processing

## Troubleshooting

1. Memory Issues:
   - Reduce batch size
   - Use smaller model variant
   - Close other applications

2. Model Download:
   - Ensure stable internet connection
   - Check disk space (~2GB needed)

3. CUDA Errors:
   - Program works on CPU if CUDA unavailable
   - Update GPU drivers if using GPU

## Model Information

BART (facebook/bart-base):
- Smaller version of the BART model (~560MB)
- Pre-trained on a large corpus of text
- Good for general sequence-to-sequence tasks
- Uses byte-pair encoding tokenization
- Efficient for memory-constrained environments
- Supports the same features as larger models but with faster inference