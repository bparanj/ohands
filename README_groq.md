# Groq API Test

A simple program to test connection to the Groq API and verify API key functionality.

## Purpose

This program:
1. Tests Groq API key validity
2. Verifies connection to Groq API service
3. Makes a test request using LLaMA 2 model
4. Shows response time and usage statistics

## Requirements

```bash
pip install groq
```

## Setup

Set your Groq API key as an environment variable:

```bash
export GROQ_API_KEY='your-api-key-here'
```

## Running the Test

```bash
python groq_test.py
```

## Expected Output

If successful, you should see:
1. Connection confirmation
2. Response time
3. Generated code response
4. Token usage information (if available)

## Test Features

The program tests:
- API key validation
- Connection stability
- Response generation
- Performance metrics

## Model Information

Uses the llama2-70b-4096 model with:
- Temperature: 0.7
- Max tokens: 500
- Top P: 1.0

## Troubleshooting

If you encounter errors:
1. Verify API key is correct
2. Check internet connection
3. Ensure Groq API service is available
4. Verify environment variable is set correctly
5. Check API credit balance

## Notes

- Response time is measured
- Includes error handling
- Shows token usage when available
- Uses non-streaming mode for simplicity