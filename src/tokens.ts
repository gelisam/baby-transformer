// Token tables for a given vocabulary size
type TokenTables = {
  shortNumbers: string[];
  shortLetters: string[];
  shortTokens: string[];
  numbers: string[];
  letters: string[];
  tokens: string[];
  tokenStringToIndex: { [key: string]: number };
};

// Cache for the most recently used vocabulary size
let cachedVocabSize: number | null = null;
let cachedTables: TokenTables | null = null;

// Calculate the output size based on vocabulary size
function getOutputSize(vocabSize: number): number {
  return vocabSize * 2; // vocabSize numbers + vocabSize letters
}

// Private function to get or create token tables for a given vocabulary size
function getTables(vocabSize: number): TokenTables {
  // Check if cached vocabSize matches
  if (cachedVocabSize === vocabSize && cachedTables !== null) {
    return cachedTables;
  }
  
  // Generate new tables
  const /*mut*/ shortNumbers: string[] = [];
  const /*mut*/ shortLetters: string[] = [];
  
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
  
  const /*mut*/ tokenStringToIndex: { [key: string]: number } = {};
  for (let i = 0; i < tokens.length; i++) {
    tokenStringToIndex[tokens[i]] = i;
  }
  
  const tables: TokenTables = {
    shortNumbers,
    shortLetters,
    shortTokens,
    numbers,
    letters,
    tokens,
    tokenStringToIndex
  };
  
  // Update cache
  cachedVocabSize = vocabSize;
  cachedTables = tables;
  
  return tables;
}

// Pure function exports - all take vocabSize as first parameter

function getTokenCount(vocabSize: number): number {
  return vocabSize * 2; // vocabSize numbers + vocabSize letters
}

function getNumberCount(vocabSize: number): number {
  return vocabSize;
}

function getLetterCount(vocabSize: number): number {
  return vocabSize;
}

function getTokenAtIndex(vocabSize: number, index: number): string {
  return getTables(vocabSize).tokens[index];
}

function getNumberAtIndex(vocabSize: number, index: number): string {
  return getTables(vocabSize).numbers[index];
}

function getLetterAtIndex(vocabSize: number, index: number): string {
  return getTables(vocabSize).letters[index];
}

function indexToTokenNumber(index: number): number {
  return index + 1;
}

function indexToShortTokenString(vocabSize: number, index: number): string {
  return getTables(vocabSize).shortTokens[index];
}

function indexToTokenString(vocabSize: number, index: number): string {
  return getTables(vocabSize).tokens[index];
}

function tokenNumberToIndex(tokenNum: number): number {
  return tokenNum - 1;
}

function tokenNumberToShortTokenString(vocabSize: number, tokenNum: number): string {
  return getTables(vocabSize).shortTokens[tokenNum - 1];
}

function tokenNumberToTokenString(vocabSize: number, tokenNum: number): string {
  return getTables(vocabSize).tokens[tokenNum - 1];
}

function tokenStringToIndex(vocabSize: number, token: string): number {
  return getTables(vocabSize).tokenStringToIndex[token];
}

function tokenStringToTokenNumber(vocabSize: number, token: string): number {
  return getTables(vocabSize).tokenStringToIndex[token] + 1;
}

export {
  getOutputSize,
  getTokenCount,
  getNumberCount,
  getLetterCount,
  getTokenAtIndex,
  getNumberAtIndex,
  getLetterAtIndex,
  indexToTokenNumber,
  indexToShortTokenString,
  indexToTokenString,
  tokenNumberToIndex,
  tokenNumberToShortTokenString,
  tokenNumberToTokenString,
  tokenStringToIndex,
  tokenStringToTokenNumber,
};
