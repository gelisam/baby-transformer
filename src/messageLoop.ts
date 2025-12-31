/**
 * Message Loop Pattern
 * 
 * This module defines the core types for the message loop pattern,
 * which replaces the orchestrator pattern for cross-component communication.
 */

// Base message type - all messages must have a type field
export type Msg = { type: string };

// Schedule function type - used to queue messages for later processing
export type Schedule = (msg: Msg) => void;

// Message loop type - processes messages and message arrays
export type MessageLoop = (msg: Msg | Msg[]) => void;

// Extend the Window interface to include the message loop
declare global {
  interface Window {
    messageLoop: MessageLoop;
  }
}
