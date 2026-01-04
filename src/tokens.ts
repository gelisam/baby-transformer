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

// Cache for token tables keyed by vocabulary size
// Using `const /*mut*/` to convey that while the reference never changes,
// the contents of the Map will be mutated over time (see AGENTS.md)
const /*mut*/ tablesCache: Map<number, TokenTables> = new Map();

// Private function to get or create token tables for a given vocabulary size
// This function caches results to avoid recreating tables for the same vocabSize
function getTables(vocabSize: number): TokenTables {
  // Check cache first
  const cached = tablesCache.get(vocabSize);
  if (cached) {
    return cached;
  }
  
  // Generate new tables
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
  
  const tables: TokenTables = {
    shortNumbers,
    shortLetters,
    shortTokens,
    numbers,
    letters,
    tokens,
    tokenStringToIndex
  };
  
  // Cache the result
  tablesCache.set(vocabSize, tables);
  
  return tables;
}

// Pure function exports - all take vocabSize as first parameter

function getNumbers(vocabSize: number): string[] {
  return getTables(vocabSize).numbers;
}

function getLetters(vocabSize: number): string[] {
  return getTables(vocabSize).letters;
}

function getTokens(vocabSize: number): string[] {
  return getTables(vocabSize).tokens;
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
  getNumbers,
  getLetters,
  getTokens,
  indexToTokenNumber,
  indexToShortTokenString,
  indexToTokenString,
  tokenNumberToIndex,
  tokenNumberToShortTokenString,
  tokenNumberToTokenString,
  tokenStringToIndex,
  tokenStringToTokenNumber,
};
