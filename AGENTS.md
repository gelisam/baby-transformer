# Coding Patterns for This Codebase

This document describes the coding patterns used in this codebase. When modifying or extending the code, please follow these patterns.

### Instead of:
```typescript
// components/a.ts
let data: Data;
export function getData() { return data; }

// components/b.ts
import { getData } from "./components/a.js";
const data = getData(); // pulls data
```

### Do this:
1. Use the orchestrator pattern for cross-component communication (see `.github/orchestrator.instructions.md` for details)
2. Don't use OOP-style getters and setters; push changes via orchestrators
3. Divide work into pure core (computation) and imperative shell (side effects, including calling orchestrator functions)
4. Each component manages its own state using component-local variables

```typescript
// components/a.ts
function computeData(): Data { ... }
function refreshData() {
  const data = computeData();
  window.setData(data); // pushes data
}

// components/b.ts
let data: Data;
const setData: SetData = (newData) => {
  data = newData; // stores locally
};
```
