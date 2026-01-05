function transposeArray(matrix: number[][]): number[][] {
  if (matrix.length === 0 || !matrix[0]) {
    return [];
  }
  const /*mut*/ transposed: number[][] = [];
  for (let i = 0; i < matrix[0].length; i++) {
    transposed.push([]);
    for (let j = 0; j < matrix.length; j++) {
      transposed[i].push(matrix[j][i]);
    }
  }
  return transposed;
}

export { transposeArray };
