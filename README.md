# Baby Transformer - TensorFlow.js Linear Regression Demo

A TypeScript-based demo that trains a simple linear regression model using TensorFlow.js to learn the equation y = 2x - 1.

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

Click the "Train Model" button to train the linear regression model. The visualization will show:
- **Blue dots**: Training data points
- **Red line**: Model's learned equation
- **Green dashed line**: True equation (y = 2x - 1)

The model should learn parameters close to slope=2 and intercept=-1.
