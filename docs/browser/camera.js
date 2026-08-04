/**
 * Where the camera belongs when the viewer's axes are rebuilt.
 *
 * Pure array logic, deliberately free of any Neuroglancer import, so it can be
 * exercised under `node --test` without a browser. `viewer.js` owns the reason
 * it exists and every call into Neuroglancer.
 */

/**
 * Carry a camera position across a change of coordinate space.
 *
 * Neuroglancer remaps a position by dimension *index* when the global
 * coordinate space is rebuilt. That space is assembled from the layers as they
 * load, so its dimension order is not the order the state asked for: a camera
 * centred while the axes were `t, z, y, x` is reinterpreted when the space
 * settles into `x, y, z, t`, and each coordinate comes to mean a different
 * axis. The camera then sits somewhere the data is not - every layer loaded,
 * nothing rendered.
 *
 * Matching on name is what the index remap was standing in for. An axis that
 * is genuinely new, or one that still lands outside the data, falls back to
 * the middle of the volume.
 *
 * Returns the corrected position, or `null` when the camera is already right.
 */
export function carryCameraOver({
  names,
  position,
  lowerBounds,
  upperBounds,
  previousNames,
  previousPosition,
}) {
  const next = Array.from(position);
  let changed = false;

  // An axis with no finite extent has no meaningful centre; leave it be.
  const centreOf = (index) => {
    const lower = lowerBounds[index];
    const upper = upperBounds[index];
    return Number.isFinite(lower) && Number.isFinite(upper)
      ? (lower + upper) / 2
      : next[index];
  };

  // Each axis keeps the coordinate it had under the same name. An axis that
  // was not there before starts in the middle of the data.
  if (previousNames && previousPosition) {
    for (let i = 0; i < names.length; i += 1) {
      const before = previousNames.indexOf(names[i]);
      const value = before === -1 ? centreOf(i) : previousPosition[before];
      if (value !== next[i]) {
        next[i] = value;
        changed = true;
      }
    }
  }

  // Whatever the history, the camera has to end up on the data.
  for (let i = 0; i < names.length; i += 1) {
    const lower = lowerBounds[i];
    const upper = upperBounds[i];
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) continue;
    if (next[i] < lower || next[i] > upper) {
      next[i] = centreOf(i);
      changed = true;
    }
  }

  return changed ? next : null;
}
