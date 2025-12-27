# Coding Patterns for This Codebase

This document describes the coding patterns used in this codebase. When modifying or extending the code, please follow these patterns.

## 1. Orchestrator Pattern for Cross-Module Communication

When modules need to communicate or share data, use the orchestrator pattern instead of direct imports between modules.

### How it works:
1. Define the orchestrator function type in `src/orchestrators/myOrchestrator.ts`
2. Implement `window.myOrchestrator()` in `main.ts` which calls all module implementations in sequence
3. Each module that needs to react to the event exports its own implementation of the function
4. Any module can call `window.myOrchestrator()` to trigger all implementations
5. Modules only import from `orchestrators/*.ts`, not from each other or from `main.ts`

### Why:
- Decouples modules from each other
- Each module manages its own state independently
- Clear, explicit data flow between modules

## 2. No OOP-Style Getters and Setters

Don't use OOP-style getters and setters to expose module-local state. Instead, push changes to other modules using orchestrator functions.

### Instead of:
```typescript
// module-a.ts
let data: Data;
export function getData() { return data; }

// module-b.ts
import { getData } from "./module-a.js";
const data = getData(); // pulls data
```

### Do this:
```typescript
// module-a.ts
function computeData(): Data { ... }
function refreshData() {
  const data = computeData();
  window.setData(data); // pushes data
}

// module-b.ts
let data: Data;
const setData: SetData = (newData) => {
  data = newData; // stores locally
};
```

### Why:
- Other modules can store the values in module-local variables
- When those modules need to read the value, they read from their own module-local variable
- Clearer data flow and ownership

## 3. Pure Core, Imperative Shell

When possible, divide work into two functions:
1. **Pure core**: A pure function that computes and returns a result without side effects
2. **Imperative shell**: A wrapper function that calls the pure function and handles side effects (like calling orchestrators)

### Example:
```typescript
// Pure core: computes the result
function generateData(): TrainingData {
  // ... computation ...
  return { inputArray, outputArray, inputTensor, outputTensor };
}

// Imperative shell: handles side effects
function refreshTrainingData(): void {
  const data = generateData();
  window.setTrainingData(data.inputArray, data.outputArray, data.inputTensor, data.outputTensor);
}
```

### Why:
- Pure functions are easier to test and reason about
- Side effects are isolated and explicit
- Better separation of concerns

## 4. Module-Local State

Each module should manage its own state using module-local variables. State should not be shared across modules directly.

### Example:
```typescript
// model.ts - owns model state
let model: Sequential | null = null;
let isTraining = false;
let currentEpoch = 0;

// viz.ts - owns visualization state
let vizInputArray: number[][] = [];
let trainingInputArray: number[][] = [];
```

### Why:
- Clear ownership of state
- Modules are self-contained
- State changes are explicit via orchestrators

## 5. Direct DOM Access During Initialization

Each module should call `document.getElementById()` directly during its initialization, rather than receiving DOM elements from `main.ts`.

### Example:
```typescript
// viz.ts
let outputCanvas: HTMLCanvasElement | null = null;
let domInitialized = false;

function initVizDom() {
  if (domInitialized) return;
  outputCanvas = document.getElementById('output-canvas') as HTMLCanvasElement;
  domInitialized = true;
}
```

### Why:
- Modules are self-contained
- Clear which DOM elements each module needs
- No need to pass DOM elements through function parameters
