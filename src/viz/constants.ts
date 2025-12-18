// Constants for visualization layout
const VIZ_ROWS = 2;
const VIZ_COLUMNS = 3;
const VIZ_EXAMPLES_COUNT = VIZ_ROWS * VIZ_COLUMNS;

// Network architecture drawing constants
const NETWORK_LAYER_HEIGHT = 20;
const NETWORK_LAYER_GAP_Y = 40;
const NETWORK_START_Y = 30;
const NETWORK_MAX_WIDTH_RATIO = 0.4;
const NETWORK_ARROW_HEAD_SIZE = 8;

// Colors for different layer types
const LAYER_COLORS = {
  embedding: '#90EE90',
  hidden: '#4682B4',
  linear: '#DDA0DD',
  output: 'rgba(255, 165, 0, 1)',
  primary: 'darkblue'
} as const;

export {
  VIZ_ROWS,
  VIZ_COLUMNS,
  VIZ_EXAMPLES_COUNT,
  NETWORK_LAYER_HEIGHT,
  NETWORK_LAYER_GAP_Y,
  NETWORK_START_Y,
  NETWORK_MAX_WIDTH_RATIO,
  NETWORK_ARROW_HEAD_SIZE,
  LAYER_COLORS
};
