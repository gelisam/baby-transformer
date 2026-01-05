# Coding Patterns for This Codebase

This document describes the coding patterns used in this codebase. When modifying or extending the code, please follow these patterns.

## Mutable Variables

Use `const /*mut*/` to convey that while a variable reference is never reassigned, its contents will change over time:

```typescript
// The Map reference never changes, but entries are added/removed
const /*mut*/ cache: Map<string, Data> = new Map();

// The array reference never changes, but elements are pushed/removed
const /*mut*/ items: Item[] = [];
```

This practice helps distinguish between:
- Truly immutable constants (use `const`)
- Variables that get reassigned (use `let`)
- Variables whose reference is constant but contents are mutable (use `const /*mut*/`)

## Message Loop Pattern

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
1. Use the message loop pattern for cross-component communication (see `.github/messageLoop.instructions.md` for details)
2. Don't use OOP-style getters and setters; push changes via messages
3. Divide work into pure core (computation) and imperative shell (side effects, including scheduling messages)
4. Each component manages its own state using component-local variables

```typescript
// components/a.ts
function computeData(): Data { ... }
function refreshData(schedule: Schedule) {
  const data = computeData();
  schedule({ type: "SetData", data } as SetDataMsg); // schedules message
}

// components/b.ts
let data: Data;
const setData: SetDataHandler = (_schedule, newData) => {
  data = newData; // stores locally
};
```
