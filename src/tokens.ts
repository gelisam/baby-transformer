// Generate tokens based on vocabulary size
function generateTokens(vocabSize: number) {
  const shortNumbers: string[] = [];
  const shortLetters: string[] = [];
  
  for (let i = 1; i <= vocabSize; i++) {
    shortNumbers.push(i.toString());
  }
  
  for (let i = 0; i < vocabSize; i++) {
    shortLetters.push(String.fromCharCode(65 + i)); // 65 is 'A'
  }
  
  const shortTokens = [...shortNumbers, ...shortLetters];
  const numbers = shortNumbers.map(n => n + " ");
  const letters = shortLetters.map(l => l + "=");
  const tokens = [...numbers, ...letters];
  
  const tokenStringToIndex: { [key: string]: number } = {};
  for (let i = 0; i < tokens.length; i++) {
    tokenStringToIndex[tokens[i]] = i;
  }
  
  return { 
    SHORT_NUMBERS: shortNumbers, 
    SHORT_LETTERS: shortLetters, 
    SHORT_TOKENS: shortTokens, 
    NUMBERS: numbers, 
    LETTERS: letters, 
    TOKENS: tokens, 
    TOKEN_STRING_TO_INDEX: tokenStringToIndex 
  };
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
