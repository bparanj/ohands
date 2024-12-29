# Data Preprocessing Demo

This program demonstrates comprehensive data preprocessing techniques for both text and structured data, commonly used in machine learning pipelines.

## Features

### 1. Text Preprocessing
- Basic cleaning (lowercasing, special character removal)
- Stop word removal
- Lemmatization
- Stemming
- Text statistics calculation
- Tokenization (word and sentence level)

### 2. Structured Data Preprocessing
- Missing value handling
- Categorical variable encoding
- Numeric feature scaling
- Data type handling

## Components

### TextPreprocessor Class
Handles text data preprocessing with methods for:
- Basic text cleaning
- Stop word removal
- Lemmatization
- Stemming
- Text statistics calculation

### DataPreprocessor Class
Handles structured data preprocessing with methods for:
- Missing value imputation
- Categorical encoding
- Feature scaling

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
pip install nltk pandas numpy scikit-learn
```

## Usage

Run the program:
```bash
python data_preprocessing.py
```

## Expected Output

The program demonstrates preprocessing on two types of data:

### 1. Text Data Example
Shows the transformation of text through various preprocessing steps:
- Original text
- Cleaned text (lowercase, no special characters)
- Text without stop words
- Lemmatized text
- Stemmed text
- Text statistics (word count, sentence count, etc.)

### 2. Structured Data Example
Shows the transformation of a sample dataset through:
- Missing value handling
- Categorical encoding
- Numeric feature scaling

## Preprocessing Steps Explained

### Text Preprocessing
1. **Basic Cleaning**
   - Convert to lowercase
   - Remove special characters and numbers
   - Remove extra whitespace

2. **Stop Word Removal**
   - Remove common words that don't carry significant meaning
   - Uses NLTK's English stop words list

3. **Lemmatization**
   - Reduce words to their base or dictionary form
   - Maintains meaning and context
   - Example: "running" → "run"

4. **Stemming**
   - Reduce words to their root form
   - Faster but less accurate than lemmatization
   - Example: "running" → "run"

### Structured Data Preprocessing
1. **Missing Value Handling**
   - Numeric: Fill with mean
   - Categorical: Fill with mode

2. **Categorical Encoding**
   - Convert categorical variables to numeric
   - Uses Label Encoding
   - Maintains encoding mapping for future use

3. **Numeric Scaling**
   - Standardize numeric features
   - Zero mean and unit variance
   - Important for many machine learning algorithms

## Best Practices

1. **Text Preprocessing**
   - Choose between lemmatization and stemming based on needs
   - Consider domain-specific stop words
   - Keep original data for reference

2. **Structured Data**
   - Check data types before preprocessing
   - Consider feature relationships when handling missing values
   - Document preprocessing steps for reproducibility

## Notes

- NLTK data is downloaded automatically on first run
- The program includes example data for demonstration
- All preprocessing steps are modular and can be used independently
- Preprocessing choices should be based on your specific use case