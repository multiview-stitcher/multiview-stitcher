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

/**
 * The middle of the data, for every axis with a finite extent.
 *
 * Used until the user takes the camera over. Neuroglancer places it itself on
 * the first valid coordinate space, but that can be before every layer has
 * reported its bounds - so the camera settles into a corner of the data and
 * the view looks empty. Re-centring as the bounds grow ends up on the whole.
 *
 * Returns the corrected position, or `null` when it is already centred.
 */
export function centreOnData({ names, position, lowerBounds, upperBounds }) {
  const next = Array.from(position);
  let changed = false;

  for (let i = 0; i < names.length; i += 1) {
    const lower = lowerBounds[i];
    const upper = upperBounds[i];
    if (!Number.isFinite(lower) || !Number.isFinite(upper)) continue;
    const centre = (lower + upper) / 2;
    if (centre !== next[i]) {
      next[i] = centre;
      changed = true;
    }
  }

  return changed ? next : null;
}
