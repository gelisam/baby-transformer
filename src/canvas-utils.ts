/**
 * Canvas drawing utilities for network visualization
 */

const ARROW_HEAD_SIZE = 8;
const ARROW_LINE_WIDTH = 6;
const ARROW_COLOR = 'darkblue';

/**
 * Draw a downward arrow on the canvas
 */
function drawDownwardArrow(
  ctx: CanvasRenderingContext2D,
  x: number,
  startY: number,
  endY: number
): void {
  ctx.lineWidth = ARROW_LINE_WIDTH;
  ctx.strokeStyle = ARROW_COLOR;
  ctx.fillStyle = ARROW_COLOR;

  // Draw arrow shaft
  ctx.beginPath();
  ctx.moveTo(x, startY);
  ctx.lineTo(x, endY - ARROW_HEAD_SIZE);
  ctx.stroke();

  // Draw arrow head
  ctx.beginPath();
  ctx.moveTo(x, endY);
  ctx.lineTo(x - ARROW_HEAD_SIZE, endY - ARROW_HEAD_SIZE);
  ctx.lineTo(x + ARROW_HEAD_SIZE, endY - ARROW_HEAD_SIZE);
  ctx.closePath();
  ctx.fill();
}

/**
 * Clear a canvas
 */
function clearCanvas(canvas: HTMLCanvasElement): void {
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}

export { drawDownwardArrow, clearCanvas };
