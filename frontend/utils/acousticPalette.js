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
  { at: 0.0, color: { r: 10, g: 18, b: 44 } },
  { at: 0.16, color: { r: 25, g: 66, b: 148 } },
  { at: 0.34, color: { r: 32, g: 140, b: 214 } },
  { at: 0.5, color: { r: 102, g: 188, b: 183 } },
  { at: 0.66, color: { r: 223, g: 212, b: 96 } },
  { at: 0.84, color: { r: 236, g: 132, b: 52 } },
  { at: 1.0, color: { r: 177, g: 38, b: 28 } },
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
