# Table of Contents

## 1. Token Operations with Hugging Face Transformers
- [`token_demo.py`](token_demo.py): Basic token operations demo
- [`README.md`](README.md): Instructions for token operations

## 2. Sequence and Summarization
- [`sequence_summary.py`](sequence_summary.py): Text sequence and summarization demo
- [`README_sequence.md`](README_sequence.md): Guide for sequence operations

## 3. Data Preprocessing
- [`data_preprocessing.py`](data_preprocessing.py): Comprehensive preprocessing demo
- [`README_preprocessing.md`](README_preprocessing.md): Data preprocessing documentation

## 4. Model Variants and Analysis
### 4.1 Analysis Metrics
- [`model_analysis_metrics.py`](model_analysis_metrics.py): Model analysis and metrics
- Part of [`README_demos.md`](README_demos.md): Analysis metrics documentation

### 4.2 Generation Parameters
- [`generation_parameters_demo.py`](generation_parameters_demo.py): Parameter exploration
- Part of [`README_demos.md`](README_demos.md): Generation parameters guide

### 4.3 Visualization
- [`visualization_demo.py`](visualization_demo.py): Interactive visualizations
- Part of [`README_demos.md`](README_demos.md): Visualization documentation

## 5. API Connection Tests
### 5.1 DeepSeek Coder
- [`deepseek_test.py`](deepseek_test.py): DeepSeek API connection test
- [`README_deepseek.md`](README_deepseek.md): DeepSeek setup and usage guide

### 5.2 Groq
- [`groq_test.py`](groq_test.py): Groq API connection test
- [`README_groq.md`](README_groq.md): Groq setup and usage guide

## File Structure
```
/workspace/
├── token_demo.py
├── README.md
├── sequence_summary.py
├── README_sequence.md
├── data_preprocessing.py
├── README_preprocessing.md
├── model_analysis_metrics.py
├── generation_parameters_demo.py
├── visualization_demo.py
├── README_demos.md
├── deepseek_test.py
├── README_deepseek.md
├── groq_test.py
├── README_groq.md
└── TABLE_OF_CONTENTS.md
```

## Key Features by Section

### 1. Token Operations
- Basic tokenization
- Token-to-ID conversion
- Special token handling
- Vocabulary exploration

### 2. Sequence and Summarization
- Text sequence processing
- BART model usage
- Text summarization
- Token sequence analysis

### 3. Data Preprocessing
- Text cleaning
- Tokenization
- Stop word removal
- Structured data handling

### 4. Model Variants and Analysis
- Model comparison
- Parameter exploration
- Performance visualization
- Metrics analysis

### 5. API Connection Tests
- API key validation
- Connection testing
- Response verification
- Error handling

## Dependencies
Each section requires specific Python packages:
```bash
# Core dependencies
pip install transformers torch pandas numpy

# Additional for preprocessing
pip install nltk scikit-learn

# For visualization
pip install plotly seaborn matplotlib

# For API tests
pip install openai groq
```