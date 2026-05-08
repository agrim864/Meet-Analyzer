const MODE_THRESHOLDS = {
  unknown: { pixelDelta: 18, minChangedRatio: 0.012 },
  other: { pixelDelta: 18, minChangedRatio: 0.012 },
  slide: { pixelDelta: 20, minChangedRatio: 0.009 },
  code: { pixelDelta: 14, minChangedRatio: 0.0025 },
  demo: { pixelDelta: 16, minChangedRatio: 0.004 },
};

let prevGray = null;
let prevMode = "unknown";

function rgbaToGray(rgba, width, height) {
  const gray = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 1) {
    gray[j] = Math.round((rgba[i] * 0.299) + (rgba[i + 1] * 0.587) + (rgba[i + 2] * 0.114));
  }
  return gray;
}

self.onmessage = (event) => {
  const { type, payload } = event.data || {};
  if (type === "reset") {
    prevGray = null;
    prevMode = payload?.mode || "unknown";
    return;
  }

  if (type !== "diff") {
    return;
  }

  const { width, height, rgbaBuffer, mode } = payload;
  const rgba = new Uint8ClampedArray(rgbaBuffer);
  const gray = rgbaToGray(rgba, width, height);
  const modeKey = MODE_THRESHOLDS[mode] ? mode : "unknown";
  const thresholds = MODE_THRESHOLDS[modeKey];

  if (!prevGray || prevGray.length !== gray.length || prevMode !== mode) {
    prevGray = gray;
    prevMode = mode;
    self.postMessage({
      changed: true,
      changedPixelRatio: 1,
      changedBboxes: [[0, 0, width, height]],
      mergedChangedBbox: [0, 0, width, height],
    });
    return;
  }

  let changed = 0;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const delta = Math.abs(gray[idx] - prevGray[idx]);
      if (delta >= thresholds.pixelDelta) {
        changed += 1;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  prevGray = gray;
  prevMode = mode;

  const changedPixelRatio = changed / Math.max(1, width * height);
  const isChanged = changedPixelRatio >= thresholds.minChangedRatio;
  const bbox = maxX >= 0 ? [minX, minY, maxX + 1, maxY + 1] : null;

  self.postMessage({
    changed: isChanged,
    changedPixelRatio,
    changedBboxes: bbox ? [bbox] : [],
    mergedChangedBbox: bbox,
  });
};
