// Generate tokens based on vocabulary size
function generateTokens(vocabSize: number) {
  const SHORT_NUMBERS: string[] = [];
  const SHORT_LETTERS: string[] = [];
  
  for (let i = 1; i <= vocabSize; i++) {
    SHORT_NUMBERS.push(i.toString());
  }
  
  for (let i = 0; i < vocabSize; i++) {
    SHORT_LETTERS.push(String.fromCharCode(65 + i)); // 65 is 'A'
  }
  
  const SHORT_TOKENS = [...SHORT_NUMBERS, ...SHORT_LETTERS];
  const NUMBERS = SHORT_NUMBERS.map(n => n + " ");
  const LETTERS = SHORT_LETTERS.map(l => l + "=");
  const TOKENS = [...NUMBERS, ...LETTERS];
  
  const TOKEN_STRING_TO_INDEX: { [key: string]: number } = {};
  for (let i = 0; i < TOKENS.length; i++) {
    TOKEN_STRING_TO_INDEX[TOKENS[i]] = i;
  }
  
  return { SHORT_NUMBERS, SHORT_LETTERS, SHORT_TOKENS, NUMBERS, LETTERS, TOKENS, TOKEN_STRING_TO_INDEX };
}

// Default vocabulary size
let currentVocabSize = 3;
let tokenData = generateTokens(currentVocabSize);

let SHORT_NUMBERS = tokenData.SHORT_NUMBERS;
let SHORT_LETTERS = tokenData.SHORT_LETTERS;
let SHORT_TOKENS = tokenData.SHORT_TOKENS;
let NUMBERS = tokenData.NUMBERS;
let LETTERS = tokenData.LETTERS;
let TOKENS = tokenData.TOKENS;
let TOKEN_STRING_TO_INDEX = tokenData.TOKEN_STRING_TO_INDEX;

// Function to update tokens when vocabulary size changes
function setVocabSize(vocabSize: number): void {
  currentVocabSize = vocabSize;
  tokenData = generateTokens(vocabSize);
  SHORT_NUMBERS = tokenData.SHORT_NUMBERS;
  SHORT_LETTERS = tokenData.SHORT_LETTERS;
  SHORT_TOKENS = tokenData.SHORT_TOKENS;
  NUMBERS = tokenData.NUMBERS;
  LETTERS = tokenData.LETTERS;
  TOKENS = tokenData.TOKENS;
  TOKEN_STRING_TO_INDEX = tokenData.TOKEN_STRING_TO_INDEX;
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
  setVocabSize,
};
