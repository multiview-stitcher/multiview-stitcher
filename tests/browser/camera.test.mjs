/**
 * The camera correction the embedded viewer depends on.
 *
 * This is a platform boundary, not an algorithm: Neuroglancer rebuilds its
 * global coordinate space from the layers as they load, and remaps the camera
 * by dimension index while doing so. When the rebuilt space lists the axes in
 * a different order than the state asked for, every coordinate silently comes
 * to mean a different axis. The only symptom is a viewer that shows nothing
 * while reporting every layer loaded - which is exactly what switching back to
 * a transform key used to do.
 *
 *   node --test tests/browser/camera.test.mjs
 */

import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

const { carryCameraOver } = await import(
  join(repoRoot, "docs", "browser", "camera.js")
);

/** The two example tiles, as the viewer sees them once registered. */
const bounds = {
  "t,z,y,x": { lowerBounds: [0, 0, 0, 0], upperBounds: [1, 12, 69, 112] },
  "x,y,z,t": { lowerBounds: [0, 0, 0, 0], upperBounds: [112, 69, 12, 1] },
};

test("a reordered space keeps every axis pointing where it did", () => {
  // The camera was centred while the axes were t,z,y,x; the space then
  // settles into x,y,z,t and Neuroglancer hands the coordinates straight
  // over, so z would read 32 - outside a volume 12 deep.
  const next = carryCameraOver({
    names: ["x", "y", "z", "t"],
    position: [0, 6, 32, 32],
    ...bounds["x,y,z,t"],
    previousNames: ["t", "z", "y", "x"],
    previousPosition: [0, 6, 32, 32],
  });

  assert.deepEqual(next, [32, 32, 6, 0]);
});

test("an unchanged space leaves the camera alone", () => {
  // Panning must not be fought: same axes, same position, nothing to do.
  const next = carryCameraOver({
    names: ["x", "y", "z", "t"],
    position: [10, 20, 3, 0],
    ...bounds["x,y,z,t"],
    previousNames: ["x", "y", "z", "t"],
    previousPosition: [10, 20, 3, 0],
  });

  assert.equal(next, null);
});

test("a position outside the data is pulled back onto it", () => {
  // No history at all - the camera simply is not on the data.
  const next = carryCameraOver({
    names: ["x", "y", "z", "t"],
    position: [10, 20, 400, 0],
    ...bounds["x,y,z,t"],
    previousNames: null,
    previousPosition: null,
  });

  assert.deepEqual(next, [10, 20, 6, 0]);
});

test("only the axes that left the data are moved", () => {
  const next = carryCameraOver({
    names: ["x", "y", "z", "t"],
    position: [10, 20, -5, 0],
    ...bounds["x,y,z,t"],
    previousNames: ["x", "y", "z", "t"],
    previousPosition: [10, 20, -5, 0],
  });

  // x and y are where the user put them; only z is corrected.
  assert.deepEqual(next, [10, 20, 6, 0]);
});

test("a new axis starts in the middle of the data", () => {
  // Adding the fused preview can introduce a dimension the views did not
  // contribute; there is no previous coordinate to carry over.
  const next = carryCameraOver({
    names: ["x", "y", "z", "t"],
    position: [10, 20, 3, 0],
    ...bounds["x,y,z,t"],
    previousNames: ["x", "y", "z"],
    previousPosition: [10, 20, 3],
  });

  assert.deepEqual(next, [10, 20, 3, 0.5]);
});

test("an axis with no finite extent is left untouched", () => {
  const next = carryCameraOver({
    names: ["x", "y"],
    position: [10, 20],
    lowerBounds: [0, Number.NEGATIVE_INFINITY],
    upperBounds: [112, Number.POSITIVE_INFINITY],
    previousNames: null,
    previousPosition: null,
  });

  assert.equal(next, null);
});

test("a dropped axis does not shift the ones that remain", () => {
  // Removing the last view can take a dimension with it; the axes that stay
  // must not slide into the freed slots.
  const next = carryCameraOver({
    names: ["x", "y", "z"],
    position: [10, 20, 3],
    lowerBounds: [0, 0, 0],
    upperBounds: [112, 69, 12],
    previousNames: ["t", "x", "y", "z"],
    previousPosition: [0, 44, 55, 6],
  });

  assert.deepEqual(next, [44, 55, 6]);
});

const { centreOnData } = await import(
  join(repoRoot, "docs", "browser", "camera.js")
);

test("an unplaced camera is centred on the whole of the data", () => {
  // Neuroglancer places the camera on the first valid coordinate space, which
  // can be before every layer has reported its bounds - leaving the view in a
  // corner of the data. Re-centring as the bounds grow ends up on the whole.
  const next = centreOnData({
    names: ["x", "y", "z", "t"],
    position: [3.4, 1.9, 1.9, 0],
    lowerBounds: [500, 480, 0, 0],
    upperBounds: [620, 564, 64, 1],
  });

  assert.deepEqual(next, [560, 522, 32, 0.5]);
});

test("centring reports no change when it is already centred", () => {
  assert.equal(
    centreOnData({
      names: ["x", "y"],
      position: [56, 34.5],
      lowerBounds: [0, 0],
      upperBounds: [112, 69],
    }),
    null,
  );
});

test("centring leaves an axis with no finite extent alone", () => {
  const next = centreOnData({
    names: ["x", "y"],
    position: [10, 20],
    lowerBounds: [0, Number.NEGATIVE_INFINITY],
    upperBounds: [112, Number.POSITIVE_INFINITY],
  });

  // Only x moves; y has no meaningful middle.
  assert.deepEqual(next, [56, 20]);
});
