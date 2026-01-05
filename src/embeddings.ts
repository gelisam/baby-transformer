import { transposeArray } from "./array.js";
import { tokenNumberToIndex } from "./tokens.js";

const EMBEDDING_DIM = 3;

// Generate embedding matrix dynamically based on vocab size
function generateEmbeddingMatrix(vocabSize: number): number[][] {
  const /*mut*/ matrix: number[][] = [];
  
  // For each number token (1, 2, 3, ...)
  for (let i = 0; i < vocabSize; i++) {
    const angle = (i * 2 * Math.PI) / vocabSize;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    matrix.push([10, cos, sin]);
  }
  
  // For each letter token (A=, B=, C=, ...)
  for (let i = 0; i < vocabSize; i++) {
    const angle = (i * 2 * Math.PI) / vocabSize;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    matrix.push([cos, 10, sin]);
  }
  
  return matrix;
}

// Default embedding matrix for vocab size 3 (for backward compatibility)
const EMBEDDING_MATRIX: number[][] = generateEmbeddingMatrix(3);
const UNEMBEDDING_MATRIX = transposeArray(EMBEDDING_MATRIX);

function embedTokenNumber(tokenNum: number, embeddingMatrix: number[][]): number[] {
  const tokenIndex = tokenNumberToIndex(tokenNum);
  return embeddingMatrix[tokenIndex];
}

function embedInput(input: number[], embeddingMatrix: number[][]): number[] {
  const /*mut*/ embeddedInput: number[] = [];
  for (let i = 0; i < input.length; i++) {
    const embedding = embedTokenNumber(input[i], embeddingMatrix);
    embeddedInput.push(...embedding);
  }
  return embeddedInput;
}

export {
  EMBEDDING_DIM,
  EMBEDDING_MATRIX,
  UNEMBEDDING_MATRIX,
  generateEmbeddingMatrix,
  embedTokenNumber,
  embedInput
};
