# DeepSeek Coder API Test

A simple program to test connection to the DeepSeek Coder API.

## Purpose

This program:
1. Tests API key validity
2. Verifies connection to DeepSeek API
3. Makes a simple test request
4. Shows usage information

## Requirements

```bash
pip install openai
```

## Setup

Set your DeepSeek API key as an environment variable:

```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

## Running the Test

```bash
python deepseek_test.py
```

## Expected Output

If successful, you should see:
1. Connection confirmation
2. Generated code response
3. Token usage information

## Troubleshooting

If you encounter errors:
1. Verify API key is correct
2. Check internet connection
3. Ensure DeepSeek API service is available
4. Verify environment variable is set correctly

## Notes

- Uses the deepseek-coder-33b-instruct model
- Makes a simple code generation request
- Shows token usage statistics
- Includes error handling and diagnostics