---
applyTo: "src/messages/*.ts"
description: a simple architecture for cross-module events using the message loop pattern
---
In this codebase, we use a message loop pattern to allow modules to interact without getting tangled.
1. Message types are defined in dedicated `messages/myMessage.ts` modules (e.g., `MyMessageMsg`), along with handler types (e.g., `MyMessageHandler`).
2. The `messageLoop.ts` module defines core types: `Msg`, `Schedule`, and `MessageLoop`, as well as the type of `window.messageLoop`.
3. Any module may define its version of the handler function (e.g., `myHandler: MyMessageHandler`), and the `main.ts` `processMessage()` function will call relevant implementations in an order which makes sense.
4. DOM event handlers call `window.messageLoop({type: "MyMessage", ...} as MyMessageMsg)` to trigger message processing.
5. Message handlers receive a `schedule` parameter which they use to queue follow-up messages: `schedule({type: "AnotherMessage", ...} as AnotherMessageMsg)`.
6. In order to do those things, the modules only have to import message types and `messageLoop.ts`, they don't have to import each other nor to import `main.ts`.
