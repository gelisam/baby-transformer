function transposeArray(matrix: number[][]): number[][] {
  const transposed: number[][] = [];
  for (let i = 0; i < matrix[0].length; i++) {
    transposed.push([]);
    for (let j = 0; j < matrix.length; j++) {
      transposed[i].push(matrix[j][i]);
    }
  }
  return transposed;
}

export { transposeArray };
