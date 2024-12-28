# Model Variants Explorer

This program demonstrates how to load and compare different variants of language models, specifically focusing on GPT-2 variants. It shows differences in their generation capabilities, performance, and resource requirements.

## Features

1. Model Management
   - Load multiple model variants
   - Compare model sizes and architectures
   - Track model statistics

2. Text Generation
   - Generate text with different models
   - Control generation parameters
   - Compare generation quality

3. Performance Analysis
   - Track generation times
   - Compare model sizes
   - Analyze resource usage

## Model Variants Included

1. **GPT-2 (124M parameters)**
   - Base model
   - Good balance of performance and size
   - Suitable for general text generation

2. **GPT-2 Medium (355M parameters)**
   - Larger model
   - Better text quality
   - Requires more resources

3. **DistilGPT2 (82M parameters)**
   - Distilled version
   - Faster inference
   - Smaller memory footprint

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
pip install transformers torch pandas
```

### 3. Run the Program

```bash
python model_variants.py
```

## Program Components

### ModelExplorer Class

1. **Model Loading**
   - Loads models and tokenizers
   - Handles errors gracefully
   - Tracks loaded models

2. **Text Generation**
   - Configurable parameters
   - Multiple generation strategies
   - Performance tracking

3. **Model Comparison**
   - Compare generation quality
   - Track performance metrics
   - Generate comparison reports

## Generation Parameters

- `max_length`: Maximum length of generated text
- `temperature`: Controls randomness (higher = more random)
- `top_k`: Limits vocabulary for next token selection
- `top_p`: Nucleus sampling parameter
- `num_samples`: Number of generations per prompt

## Output Analysis

The program generates:
1. Generated text samples
2. Performance metrics
3. Model statistics
4. CSV file with detailed results

## Memory Requirements

- GPT-2: ~500MB
- GPT-2 Medium: ~1.5GB
- DistilGPT2: ~300MB

## Best Practices

1. **Model Selection**
   - Choose based on available resources
   - Consider speed vs quality tradeoff
   - Test with specific use cases

2. **Generation Parameters**
   - Adjust temperature for creativity
   - Use top_k/top_p for better quality
   - Balance length with resource usage

3. **Resource Management**
   - Load models sequentially
   - Clear unused models
   - Monitor memory usage

## Troubleshooting

1. Memory Issues
   - Use smaller models
   - Reduce batch size
   - Clear model cache

2. Generation Quality
   - Adjust temperature
   - Modify prompt
   - Try different models

3. Performance
   - Use GPU if available
   - Reduce max_length
   - Consider distilled models

## Notes

- First run downloads models
- GPU acceleration if available
- Results vary by hardware