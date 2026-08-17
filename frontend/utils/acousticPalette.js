function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mixChannel(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function mixColor(a, b, t) {
  return {
    r: mixChannel(a.r, b.r, t),
    g: mixChannel(a.g, b.g, t),
    b: mixChannel(a.b, b.b, t),
  };
}

function rgbToCss({ r, g, b }, alpha = 1) {
  const safeAlpha = clamp(alpha, 0, 1);
  return `rgba(${r}, ${g}, ${b}, ${safeAlpha})`;
}

const stops = [
  { at: 0.0, color: { r: 8, g: 10, b: 24 } },
  { at: 0.24, color: { r: 13, g: 32, b: 78 } },
  { at: 0.5, color: { r: 0, g: 238, b: 255 } },
  { at: 0.72, color: { r: 218, g: 70, b: 255 } },
  { at: 1.0, color: { r: 255, g: 117, b: 24 } },
];

export function sampleAcousticPalette(value, alpha = 1) {
  const normalized = clamp((Number(value) + 1) * 0.5, 0, 1);
  const lower = stops.reduce((best, stop) => (stop.at <= normalized ? stop : best), stops[0]);
  const upper = stops.find((stop) => stop.at >= normalized) || stops[stops.length - 1];
  const span = Math.max(upper.at - lower.at, 1e-6);
  const t = clamp((normalized - lower.at) / span, 0, 1);
  const rgb = mixColor(lower.color, upper.color, t);

  return {
    ...rgb,
    alpha: clamp(alpha, 0, 1),
    css: rgbToCss(rgb, alpha),
  };
}

export function acousticLegendGradient(alpha = 1) {
  return stops
    .map((stop) => `${sampleAcousticPalette(stop.at * 2 - 1, alpha).css} ${Math.round(stop.at * 100)}%`)
    .join(', ');
}
