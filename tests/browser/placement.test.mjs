/**
 * The rules behind manual tile placement, with no browser.
 *
 * `docs/browser/placement.js` is deliberately free of DOM and Neuroglancer, so
 * the dimension bookkeeping a drag depends on - which tile, how far, in which
 * of three coordinate systems - can be checked here rather than only through a
 * headless page.
 *
 *   node --test tests/browser/placement.test.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const source = join(repoRoot, "docs", "browser", "placement.js");

// Imported through a data: URL rather than by path. `docs/browser/` holds
// browser ES modules with a `.js` extension and no package.json saying so -
// and it cannot have one, since `routes.js` is loaded as a classic script and
// `require`d by routes.test.mjs. Node would read this file as CommonJS and
// fail on `export`.
const {
  boundsContain,
  composeAffine,
  fromPhysicalMatrix,
  dragAngle,
  pickDragTarget,
  pixelOffset,
  planeRotation,
  rotationMatrix,
  toPhysical,
  toPhysicalMatrix,
  translateMatrix,
  translationForDrag,
} = await import(
  `data:text/javascript;base64,${readFileSync(source).toString("base64")}`
);

const close = (actual, expected, message) =>
  assert.ok(
    Math.abs(actual - expected) < 1e-9,
    `${message ?? "value"}: ${actual} != ${expected}`,
  );

const closeAll = (actual, expected, message) => {
  assert.equal(actual.length, expected.length, message);
  actual.forEach((value, i) => close(value, expected[i], `${message}[${i}]`));
};

test("the pointer decides when only one tile is under it", () => {
  assert.deepEqual(pickDragTarget(["a"], []), { urls: ["a"], reason: "only" });
  // Even when something else is selected: the pointer is unambiguous.
  assert.deepEqual(pickDragTarget(["a"], ["b"]), {
    urls: ["a"],
    reason: "only",
  });
  // One tile reported once per channel is still one tile.
  assert.deepEqual(pickDragTarget(["a", "a"], []), {
    urls: ["a"],
    reason: "only",
  });
});

test("where tiles overlap, the selection decides", () => {
  assert.deepEqual(pickDragTarget(["a", "b"], ["b"]), {
    urls: ["b"],
    reason: "selected",
  });
  assert.deepEqual(pickDragTarget(["a", "b"], ["a"]), {
    urls: ["a"],
    reason: "selected",
  });
});

test("an ambiguous drag is refused rather than guessed at", () => {
  // Moving the wrong tile is worse than moving none, and the user has a way
  // to say which they mean.
  assert.deepEqual(pickDragTarget(["a", "b"], []), {
    urls: [],
    reason: "ambiguous",
  });
  assert.deepEqual(pickDragTarget(["a", "b"], ["c"]), {
    urls: [],
    reason: "ambiguous",
  });
  assert.deepEqual(pickDragTarget([], ["a"]), { urls: [], reason: "empty" });
});

test("several selected tiles are dragged together", () => {
  // Moving a whole row of a grid at once, or correcting a stage offset shared
  // by a batch: with a multi-selection the user has already said what they
  // mean, so the pointer no longer narrows it down.
  assert.deepEqual(pickDragTarget(["a"], ["a", "b"]), {
    urls: ["a", "b"],
    reason: "selection",
  });
  // Including tiles the pointer is nowhere near.
  assert.deepEqual(pickDragTarget(["a", "c"], ["a", "b"]), {
    urls: ["a", "b"],
    reason: "selection",
  });
});

test("a multi-tile drag has to start on one of the selected tiles", () => {
  // Otherwise a stray drag over the rest of the image would shift the whole
  // selection, with nothing under the pointer to suggest it would.
  assert.deepEqual(pickDragTarget(["c"], ["a", "b"]), {
    urls: [],
    reason: "outside-selection",
  });
  assert.deepEqual(pickDragTarget([], ["a", "b"]), {
    urls: [],
    reason: "empty",
  });
});

const space = {
  names: ["t", "c'", "z", "y", "x"],
  scales: [1, 1, 1e-6, 1e-6, 1e-6],
  lowerBounds: [0, 0, 0, 0, 0],
  upperBounds: [1, 2, 10, 64, 64],
};
const globalNames = ["t", "z", "y", "x"];
const globalScales = [1, 1e-6, 1e-6, 1e-6];

test("a position inside every shared dimension is inside the tile", () => {
  assert.equal(
    boundsContain(space, [0, 5, 32, 32], globalNames, globalScales),
    true,
  );
});

test("a position past any bound is outside", () => {
  assert.equal(
    boundsContain(space, [0, 5, 32, 64], globalNames, globalScales),
    false,
  );
  assert.equal(
    boundsContain(space, [0, 5, -1, 32], globalNames, globalScales),
    false,
  );
});

test("a dimension the tile does not share places no constraint", () => {
  // `c'` is local to the layer: the global space has no channel axis, and a
  // tile is under the pointer whichever channel is on screen.
  assert.equal(
    boundsContain(space, [0, 5, 32, 32], globalNames, globalScales),
    true,
  );
  // A dimension with no finite extent is not a reason to exclude a tile.
  const unbounded = {
    ...space,
    lowerBounds: [-Infinity, 0, 0, 0, 0],
    upperBounds: [Infinity, 2, 10, 64, 64],
  };
  assert.equal(
    boundsContain(unbounded, [999, 5, 32, 32], globalNames, globalScales),
    true,
  );
});

test("bounds in the layer's own units are compared in them", () => {
  // The same physical position, in a layer whose voxels are twice the size.
  const coarse = { ...space, scales: [1, 1, 2e-6, 2e-6, 2e-6] };
  // 40 global voxels is 20 of the layer's, so still inside its 0..64.
  assert.equal(
    boundsContain(coarse, [0, 5, 40, 40], globalNames, globalScales),
    true,
  );
  // 200 global voxels is 100 of the layer's, which is past its extent.
  assert.equal(
    boundsContain(coarse, [0, 5, 200, 40], globalNames, globalScales),
    false,
  );
});

const dragGeometry = {
  // The xy panel: display axes 0 and 1 are drawn from global x and y, and the
  // third is the slice normal - which a drag never moves along.
  displayDimensionIndices: [3, 2, 1],
  globalNames,
  globalScales,
  outputNames: ["t", "c'", "z", "y", "x"],
  outputScales: [1, 1, 1e-6, 1e-6, 1e-6],
};

test("a drag moves the tile only along the axes on screen", () => {
  const translation = translationForDrag({
    ...dragGeometry,
    displayDelta: [12, -7, 0],
  });

  assert.deepEqual(translation, [0, 0, 0, -7, 12]);
});

test("a drag never moves a tile through the slice", () => {
  // Whatever the third display axis reports, the depth of a cross-section is
  // not something a drag in its plane may change.
  const translation = translationForDrag({
    ...dragGeometry,
    displayDelta: [12, -7, 0],
  });
  const zIndex = dragGeometry.outputNames.indexOf("z");

  assert.equal(translation[zIndex], 0);
});

test("a drag is rescaled into the tile's own dimensions", () => {
  const translation = translationForDrag({
    ...dragGeometry,
    outputScales: [1, 1, 2e-6, 2e-6, 2e-6],
    displayDelta: [12, -8, 0],
  });

  // Half as many of the layer's voxels cover the same distance.
  assert.deepEqual(translation, [0, 0, 0, -4, 6]);
});

test("a dimension the layer does not have is dropped, not misapplied", () => {
  // Matching by index instead of name is exactly what would put a `y` drag on
  // whatever axis happened to sit at that index.
  const translation = translationForDrag({
    ...dragGeometry,
    outputNames: ["z", "y", "x"],
    outputScales: [1e-6, 1e-6, 1e-6],
    displayDelta: [12, -7, 0],
  });

  assert.deepEqual(translation, [0, -7, 12]);
});

test("a layer whose dimensions are in another order still moves correctly", () => {
  const translation = translationForDrag({
    ...dragGeometry,
    outputNames: ["x", "y", "z", "t"],
    outputScales: [1e-6, 1e-6, 1e-6, 1],
    displayDelta: [12, -7, 0],
  });

  assert.deepEqual(translation, [12, -7, 0, 0]);
});

test("only the translation column of a transform moves", () => {
  // Column-major, stride rank + 1: a 2x2 rotation with a translation of (5, 6).
  const rank = 2;
  const matrix = Float64Array.from([0, 1, 0, -1, 0, 0, 5, 6, 1]);

  const moved = translateMatrix(matrix, rank, [3, -2]);

  assert.deepEqual(Array.from(moved), [0, 1, 0, -1, 0, 0, 8, 4, 1]);
  // The input is left alone: a drag recomputes from the same starting matrix
  // on every pointer move, so accumulating into it would drift.
  assert.deepEqual(Array.from(matrix), [0, 1, 0, -1, 0, 0, 5, 6, 1]);
});

test("the homogeneous row of a transform is never touched", () => {
  const rank = 3;
  const identity = Float64Array.from([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  ]);

  const moved = translateMatrix(identity, rank, [1, 2, 3, 99]);

  assert.equal(moved[moved.length - 1], 1);
  assert.deepEqual(Array.from(moved.slice(12, 15)), [1, 2, 3]);
});

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------

test("a display vector is measured in physical units, not voxels", () => {
  // Voxels are not cubes, and a rotation applied to voxel counts is a shear.
  assert.deepEqual(toPhysical([2, 2, 2], [1e-6, 1e-6, 4e-6]), [
    2e-6, 2e-6, 8e-6,
  ]);
});

test("a physical offset is projected back onto the panel's pixels", () => {
  // One pixel across is 1 um; one pixel down is 1 um the other way.
  const u = [1e-6, 0, 0];
  const v = [0, 1e-6, 0];

  const offset = pixelOffset([3e-6, -4e-6, 0], u, v);

  close(offset.x, 3, "x");
  close(offset.y, -4, "y");
});

test("a zoomed-in panel projects fewer physical units per pixel", () => {
  const offset = pixelOffset([3e-6, 0, 0], [0.5e-6, 0, 0], [0, 0.5e-6, 0]);
  close(offset.x, 6, "x");
});

test("the drag angle is measured around the tile's centre", () => {
  const centre = { x: 0, y: 0 };

  close(dragAngle(centre, { x: 10, y: 0 }, { x: 0, y: 10 }), Math.PI / 2);
  close(dragAngle(centre, { x: 0, y: 10 }, { x: 10, y: 0 }), -Math.PI / 2);
  // The radius does not matter, only the sweep.
  close(dragAngle(centre, { x: 10, y: 0 }, { x: 0, y: 99 }), Math.PI / 2);
  // Nor does where the centre is.
  close(
    dragAngle({ x: 5, y: 5 }, { x: 15, y: 5 }, { x: 5, y: 15 }),
    Math.PI / 2,
  );
});

test("a rotation turns the plane on screen and leaves its normal alone", () => {
  const u = [1e-6, 0, 0];
  const v = [0, 1e-6, 0];

  const quarter = planeRotation(u, v, Math.PI / 2);

  // u goes to v, v goes back to -u: the direction a positive drag angle turns.
  closeAll(
    [quarter[0][0], quarter[1][0], quarter[2][0]],
    [0, 1, 0],
    "image of u",
  );
  closeAll(
    [quarter[0][1], quarter[1][1], quarter[2][1]],
    [-1, 0, 0],
    "image of v",
  );
  // The plane normal is untouched, which is what makes this the 2D rotation
  // the user sees rather than a tumble in 3D.
  closeAll(
    [quarter[0][2], quarter[1][2], quarter[2][2]],
    [0, 0, 1],
    "image of n",
  );
});

test("a rotation is built in the panel's plane, not a coordinate plane", () => {
  // A cross-section the user has turned 45 degrees.
  const s = Math.SQRT1_2 * 1e-6;
  const rotation = planeRotation([s, s, 0], [-s, s, 0], Math.PI / 2);

  // Still a quarter turn about z, because that is this plane's normal.
  closeAll(
    [rotation[0][0], rotation[1][0], rotation[2][0]],
    [0, 1, 0],
    "image of x",
  );
});

test("a rotation about a plane the layer does not span leaves it alone", () => {
  const rank = 3;
  const identity = Float64Array.from([
    1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
  ]);

  const turn = rotationMatrix({
    rotation: planeRotation([1, 0, 0], [0, 1, 0], Math.PI / 3),
    centre: [0, 0, 0],
    axes: [-1, -1, -1],
    outputScales: [1e-6, 1e-6, 1e-6],
    rank,
  });

  closeAll(Array.from(turn), Array.from(identity), "identity");
});

/** Apply a physical homogeneous matrix to a physical point. */
const applyPhysical = (matrix, rank, point) =>
  Array.from({ length: rank }, (_, row) => {
    let sum = matrix[(rank + 1) * rank + row];
    for (let k = 0; k < rank; k += 1) {
      sum += matrix[(rank + 1) * k + row] * point[k];
    }
    return sum;
  });

test("a quarter turn about the centre swaps the axes on screen", () => {
  const rank = 2;
  const turn = rotationMatrix({
    rotation: planeRotation([1e-6, 0, 0], [0, 1e-6, 0], Math.PI / 2),
    centre: [10, 10],
    axes: [0, 1, -1],
    outputScales: [1e-6, 1e-6],
    rank,
  });

  // The centre is a fixed point: rotating about it must not move it.
  closeAll(applyPhysical(turn, rank, [10e-6, 10e-6]), [10e-6, 10e-6], "centre");
  // A point one unit along the first axis lands one unit along the second.
  closeAll(applyPhysical(turn, rank, [11e-6, 10e-6]), [10e-6, 11e-6], "turned");
});

test("a rotation's linear block does not depend on the dimension scales", () => {
  // Neuroglancer rescales a source transform's linear coefficients by the
  // input and output dimension scales itself, so they are already physical.
  // Scaling them here as well is a shear - and one that hides completely in
  // an xy view, where the two axes share a spacing.
  const rank = 2;
  const rotation = planeRotation([1e-6, 0, 0], [0, 1e-6, 0], Math.PI / 3);
  const shared = { rotation, centre: [0, 0], axes: [0, 1, -1], rank };

  const isotropic = rotationMatrix({ ...shared, outputScales: [1e-6, 1e-6] });
  const anisotropic = rotationMatrix({ ...shared, outputScales: [4e-6, 1e-6] });

  closeAll(
    [isotropic[0], isotropic[1], isotropic[3], isotropic[4]],
    [anisotropic[0], anisotropic[1], anisotropic[3], anisotropic[4]],
    "linear block",
  );
});

test("a rotation across axes of different spacing stays rigid", () => {
  const rank = 2;
  const outputScales = [4e-6, 1e-6];
  const turn = rotationMatrix({
    rotation: planeRotation([1e-6, 0, 0], [0, 1e-6, 0], Math.PI / 2),
    centre: [0, 0],
    axes: [0, 1, -1],
    outputScales,
    rank,
  });

  // A metre along the coarse axis has to land a metre along the fine one.
  // Getting this wrong stretches the tile by the ratio of the spacings, which
  // is what "the rotation shears in xz" looks like.
  closeAll(applyPhysical(turn, rank, [1e-6, 0]), [0, 1e-6], "coarse axis");
  closeAll(applyPhysical(turn, rank, [0, 1e-6]), [-1e-6, 0], "fine axis");
});

test("a matrix is converted to physical units and back unchanged", () => {
  const rank = 2;
  const outputScales = [4e-6, 1e-6];
  const matrix = Float64Array.from([0, 1, 0, -1, 0, 0, 7, -3, 1]);

  const physical = toPhysicalMatrix(matrix, rank, outputScales);
  // Only the translation moves: the linear block is physical already.
  closeAll(
    [physical[0], physical[1], physical[3], physical[4]],
    [0, 1, -1, 0],
    "linear block",
  );
  closeAll([physical[6], physical[7]], [28e-6, -3e-6], "translation");

  closeAll(
    Array.from(fromPhysicalMatrix(physical, rank, outputScales)),
    Array.from(matrix),
    "round trip",
  );
});

test("composing applies the second transform first", () => {
  const rank = 2;
  // Move by (1, 0), then turn a quarter about the origin.
  const move = Float64Array.from([1, 0, 0, 0, 1, 0, 1, 0, 1]);
  const turn = Float64Array.from([0, 1, 0, -1, 0, 0, 0, 0, 1]);

  const composed = composeAffine(turn, move, rank);

  // The origin is moved to (1, 0) and then turned onto (0, 1).
  closeAll(
    [composed[6], composed[7], composed[8]],
    [0, 1, 1],
    "image of the origin",
  );
});

test("a rotation composes onto a tile that has already been moved", () => {
  // The composition happens in physical units. Doing it on Neuroglancer's own
  // mixture instead multiplies a translation that is in output pixels by a
  // linear block that is not, which drags the centre of the turn off the tile
  // - by the ratio of the spacings, so again only outside an xy view.
  const rank = 2;
  const outputScales = [4e-6, 1e-6];
  const centre = [10, 10];
  // The tile has already been moved by 5 pixels along each axis.
  const moved = Float64Array.from([1, 0, 0, 0, 1, 0, 5, 5, 1]);
  const movedCentre = [centre[0] + 5, centre[1] + 5];

  const turn = rotationMatrix({
    rotation: planeRotation([1e-6, 0, 0], [0, 1e-6, 0], Math.PI / 2),
    centre: movedCentre,
    axes: [0, 1, -1],
    outputScales,
    rank,
  });
  const composed = fromPhysicalMatrix(
    composeAffine(turn, toPhysicalMatrix(moved, rank, outputScales), rank),
    rank,
    outputScales,
  );

  // The tile's centre is where the turn happens, so it must not move: in the
  // source's own coordinates it is still at `centre`, and it must land on the
  // moved centre it was already at.
  const physical = toPhysicalMatrix(composed, rank, outputScales);
  closeAll(
    applyPhysical(physical, rank, [centre[0] * 4e-6, centre[1] * 1e-6]),
    [movedCentre[0] * 4e-6, movedCentre[1] * 1e-6],
    "centre stays put",
  );
});
