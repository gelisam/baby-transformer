/**
 * Application configuration and constants
 */

/** Visualization layout */
export const VIZ_CONFIG = {
  rows: 2,
  columns: 3,
  sectionSpacing: 10,
  barSpacing: 3
} as const;

/** Canvas dimensions and styling */
export const CANVAS_CONFIG = {
  lossChart: {
    marginX: 30,
    marginY: 30
  },
  networkArchitecture: {
    layerHeight: 20,
    maxLayerWidthRatio: 0.4,
    layerGapY: 40,
    startY: 30,
    lineWidth: {
      connections: 2,
      layerBorder: 4,
      inputBorder: 2
    }
  }
} as const;

/** Network architecture defaults */
export const NETWORK_DEFAULTS = {
  numLayers: 4,
  neuronsPerLayer: 6
} as const;

/** Training configuration */
export const TRAINING_CONFIG = {
  epochsPerBatch: 1
} as const;

/** UI messages */
export const UI_MESSAGES = {
  ready: 'Ready to train!',
  clickToStart: 'Click the button to start training...',
  training: (epoch: number, loss: number) => 
    `Training... Epoch ${epoch} - Loss: ${loss.toFixed(4)}`
} as const;
