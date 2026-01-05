import { transposeArray } from "./array.js";
import { tokenNumberToIndex } from "./tokens.js";

const EMBEDDING_DIM = 3;

const COS0 = 1;
const SIN0 = 0;
const COS120 = -0.5;
const SIN120 = Math.sqrt(3) / 2;
const COS240 = -0.5;
const SIN240 = -Math.sqrt(3) / 2;

const EMBEDDING_MATRIX: number[][] = [
  [10, COS0, SIN0],      // "1 "
  [10, COS120, SIN120],  // "2 "
  [10, COS240, SIN240],  // "3 "
  [COS0, 10, SIN0],      // "A="
  [COS120, 10, SIN120],  // "B="
  [COS240, 10, SIN240]   // "C="
];
const UNEMBEDDING_MATRIX = transposeArray(EMBEDDING_MATRIX);

function embedTokenNumber(tokenNum: number): number[] {
  const tokenIndex = tokenNumberToIndex(tokenNum);
  return EMBEDDING_MATRIX[tokenIndex];
}

function embedInput(input: number[]): number[] {
  const embeddedInput: number[] = [];
  for (let i = 0; i < input.length; i++) {
    const embedding = embedTokenNumber(input[i]);
    embeddedInput.push(...embedding);
  }
  return embeddedInput;
}

export {
  EMBEDDING_DIM,
  EMBEDDING_MATRIX,
  UNEMBEDDING_MATRIX,
  embedTokenNumber,
  embedInput
};
