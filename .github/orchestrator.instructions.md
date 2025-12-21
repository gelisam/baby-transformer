---
applyTo: "src/orchestrators/*.ts"
description: a simple architecture for cross-module events
---
In this codebase, we use global orchestrator functions to allow modules to interact without getting tangled.
1. A global orchestrator function has a type defined in a dedicated `orchestrators/myOrchestrator.ts` module, and an `window.myOrchestrator()` implementation defined in `main.ts`.
2. Any module may define its version of the `myOrchestrator()` function, and the `main.ts` implementation will call all of those implementations in an order which makes sense. 
3. Any module may call `window.myOrchestrator()` in order to trigger all of those implementations.
4. In order to do those things, the modules only have to import `myOrchestrator.ts`, they don't have to import each other nor to import `main.ts`.
