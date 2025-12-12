# Baby Transformer - TensorFlow.js Sequence Prediction Demo

A TypeScript-based demo that trains a neural network using TensorFlow.js to learn to predict the next token in sequences where letters are consistently mapped to numbers.

## Task Description

The model learns sequences of the form "{A,B,C}={1,2,3} {A,B,C}={1,2,3} {A,B,C}={1,2,3}", where:
- Each letter (A, B, or C) is followed by an equals sign (=) and a number (1, 2, or 3)
- Within a single sequence, the same letter must always map to the same number
- The model takes the first 5 tokens as input and predicts the 6th token

## Setup

Install dependencies:
```bash
npm install
```

## Build

Compile TypeScript to JavaScript:
```bash
npm run build
```

For automatic recompilation on file changes:
```bash
npm run watch
```

## Run

Start the development server:
```bash
npm run serve
```

Then open in your browser:
```
http://localhost:8000/index.html
```

## Usage

Click the "Train Model" button to train the neural network. The visualization will show:
- **Loss Curve**: Training progress over time
- **Predictions**: 6 random input sequences (5 tokens each) and the model's probability distribution for the next token
- **Network Architecture**: The structure of the neural network

The model should learn to predict the next token based on the pattern that letters consistently map to the same numbers within each sequence.
