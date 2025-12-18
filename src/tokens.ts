const SHORT_NUMBERS = ["1", "2", "3"];
const SHORT_LETTERS = ["A", "B", "C"];
const SHORT_TOKENS = [...SHORT_NUMBERS, ...SHORT_LETTERS];
const NUMBERS = SHORT_NUMBERS.map(n => n + " ");
const LETTERS = SHORT_LETTERS.map(l => l + "=");
const TOKENS = [...NUMBERS, ...LETTERS];

const TOKEN_STRING_TO_INDEX: { [key: string]: number } = {};
for (let i = 0; i < TOKENS.length; i++) {
  TOKEN_STRING_TO_INDEX[TOKENS[i]] = i;
}

function indexToTokenNumber(index: number): number {
  return index + 1;
}

function indexToShortTokenString(index: number): string {
  return SHORT_TOKENS[index];
}

function indexToTokenString(index: number): string {
  return TOKENS[index];
}

function tokenNumberToIndex(tokenNum: number): number {
  return tokenNum - 1;
}

function tokenNumberToShortTokenString(tokenNum: number): string {
  return SHORT_TOKENS[tokenNum - 1];
}

function tokenNumberToTokenString(tokenNum: number): string {
  return TOKENS[tokenNum - 1];
}

function tokenStringToIndex(token: string): number {
  return TOKEN_STRING_TO_INDEX[token];
}

function tokenStringToTokenNumber(token: string): number {
  return TOKEN_STRING_TO_INDEX[token] + 1;
}

export {
  NUMBERS,
  LETTERS,
  TOKENS,
  indexToTokenNumber,
  indexToShortTokenString,
  indexToTokenString,
  tokenNumberToIndex,
  tokenNumberToShortTokenString,
  tokenNumberToTokenString,
  tokenStringToIndex,
  tokenStringToTokenNumber,
};
