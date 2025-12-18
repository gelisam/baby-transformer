import { INPUT_SIZE } from "./constants.js";
import { TOKENS, tokenStringToTokenNumber } from "./tokens.js";

/**
 * Parse a string input into an array of token numbers.
 * 
 * @param inputStr - The input string to parse (e.g., "A=1 B=2 C=")
 * @returns An array of token numbers if valid, null if invalid
 * 
 * @example
 * parseInputString("A=1 B=2 C=") // Returns array of token numbers
 * parseInputString("invalid") // Returns null
 */
function parseInputString(inputStr: string): number[] | null {
  const tokens: number[] = [];
  let i = 0;

  while (i < inputStr.length) {
    let matched = false;

    for (const token of TOKENS) {
      if (inputStr.substring(i, i + token.length) === token) {
        tokens.push(tokenStringToTokenNumber(token));
        i += token.length;
        matched = true;
        break;
      }
    }

    if (!matched) {
      return null;
    }
  }

  return tokens.length === INPUT_SIZE ? tokens : null;
}

export { parseInputString };
