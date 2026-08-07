/**
 * Where a layer lands on a cross-section panel, with no browser.
 *
 * `docs/browser/highlight.js` decides the shape of every layer outline the
 * viewer draws. The cases worth pinning down are the ones a screenshot would
 * not tell apart from a bug: a slice that misses the tile, a tile that runs off
 * the edge of the panel, a tile that has been turned, and a cross-section the
 * user has rotated.
 *
 *   node --test tests/browser/highlight.test.mjs
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const source = join(repoRoot, "docs", "browser", "highlight.js");

// Imported through a data: URL rather than by path; see placement.test.mjs for
// why `docs/browser/` cannot declare itself as ES modules to Node.
const {
  clipToHalfPlane,
  containsPoint,
  coveragePolygon,
  layerGeometry,
  layerSlabs,
  solve,
} = await import(
  `data:text/javascript;base64,${readFileSync(source).toString("base64")}`
);

/** The slabs a panel showing `centre + x·u + y·v` cuts from a placed layer. */
function slabsFor({ centre, u, v, matrix, translation, lower, upper }) {
  return layerSlabs(
    { matrix, translation: translation ?? matrix.map(() => 0), lower, upper },
    centre,
    u,
    v,
  );
}

/**
 * A Neuroglancer-shaped transform, as `viewer.js` hands one over.
 *
 * `linear` is the source-to-output map over the last dimensions - the spatial
 * ones - with the leading dimensions left as identity, which is the shape a
 * multiview-stitcher layer actually has: a channel or time axis that no
 * placement ever touches, and the spatial block that carries the placement.
 */
function placement({ leading = [], linear, shift, size, scales }) {
  const rank = leading.length + linear.length;
  const matrix = new Float64Array((rank + 1) * (rank + 1));
  const at = (row, column, value) => {
    matrix[(rank + 1) * column + row] = value;
  };

  for (let i = 0; i < leading.length; i += 1) at(i, i, 1);
  for (let row = 0; row < linear.length; row += 1) {
    for (let column = 0; column < linear.length; column += 1) {
      at(leading.length + row, leading.length + column, linear[row][column]);
    }
    at(leading.length + row, rank, shift?.[row] ?? 0);
  }
  at(rank, rank, 1);

  const sourceLower = [...leading.map(() => -0.5), ...size.map(() => -0.5)];
  const sourceUpper = [
    ...leading.map((n) => n - 0.5),
    ...size.map((n) => n - 0.5),
  ];

  // The axis-aligned box around the result, which is what Neuroglancer itself
  // reports and what the fallback path has to work from.
  const outputLower = sourceLower.slice();
  const outputUpper = sourceUpper.slice();
  for (let row = 0; row < linear.length; row += 1) {
    let low = Infinity;
    let high = -Infinity;
    for (let corner = 0; corner < 1 << linear.length; corner += 1) {
      let value = shift?.[row] ?? 0;
      for (let column = 0; column < linear.length; column += 1) {
        const p =
          (corner >> column) & 1
            ? size[column] - 0.5
            : -0.5;
        value += linear[row][column] * p;
      }
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
    outputLower[leading.length + row] = low;
    outputUpper[leading.length + row] = high;
  }

  const spacings = scales ?? [
    ...leading.map(() => 1),
    ...size.map(() => 1),
  ];

  return {
    rank,
    sourceRank: rank,
    matrix,
    sourceLower,
    sourceUpper,
    // The two spaces of a multiview-stitcher layer are the same grid; a
    // placement moves the image within it rather than resampling it.
    sourceScales: spacings,
    outputScales: spacings,
    outputLower,
    outputUpper,
  };
}

const turn = (angle) => [
  [Math.cos(angle), -Math.sin(angle)],
  [Math.sin(angle), Math.cos(angle)],
];

/** A panel 200x100 showing the x/y plane at one unit per pixel. */
const panel = { width: 200, height: 100 };
const flat = { centre: [0, 0], u: [1, 0], v: [0, 1] };
const identity2 = [
  [1, 0],
  [0, 1],
];

/** The axis-aligned box a polygon occupies, rounded to the pixel. */
function extent(polygon) {
  const xs = polygon.map(([x]) => x);
  const ys = polygon.map(([, y]) => y);
  return [
    Math.round(Math.min(...xs)),
    Math.round(Math.min(...ys)),
    Math.round(Math.max(...xs)),
    Math.round(Math.max(...ys)),
  ];
}

const area = (polygon) => {
  let sum = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % polygon.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
};

test("a tile inside the panel is outlined where it is", () => {
  const polygon = coveragePolygon({
    ...panel,
    slabs: slabsFor({
      ...flat,
      matrix: identity2,
      lower: [-40, -10],
      upper: [20, 30],
    }),
  });

  // Panel coordinates run from its top-left, and the centre of the panel is
  // the navigation position - so -40..20 across lands at 60..120 of 200.
  assert.deepEqual(extent(polygon), [60, 40, 120, 80]);
  assert.equal(polygon.length, 4);
});

test("a tile larger than the panel is clipped to it", () => {
  // Nothing off-screen is drawn, so the overlay never has to be clipped again
  // by whoever renders it.
  const polygon = coveragePolygon({
    ...panel,
    slabs: slabsFor({
      ...flat,
      matrix: identity2,
      lower: [-1000, -1000],
      upper: [1000, 1000],
    }),
  });

  assert.deepEqual(extent(polygon), [0, 0, 200, 100]);
});

test("a turned tile is outlined at its own angle", () => {
  // The whole point of working in the layer's own coordinates: the outline is
  // the image's four edges, not the upright box it happens to sit inside.
  const angle = Math.PI / 6;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const polygon = coveragePolygon({
    width: 300,
    height: 300,
    slabs: slabsFor({
      centre: [0, 0],
      u: [1, 0],
      v: [0, 1],
      matrix: [
        [cos, -sin],
        [sin, cos],
      ],
      lower: [-30, -20],
      upper: [30, 20],
    }),
  });

  assert.equal(polygon.length, 4);
  // Still a 60x40 rectangle, so still 2400 square pixels - it has been turned,
  // not stretched. The upright box around it is a third larger.
  assert.ok(Math.abs(area(polygon) - 60 * 40) < 1e-6, area(polygon));
  const [left, top, right, bottom] = extent(polygon);
  assert.ok(right - left > 60, "a turned tile is wider than its own width");
  assert.ok(bottom - top > 40, "and taller than its own height");
});

test("a slice that misses the tile outlines nothing", () => {
  // The depth axis has no component along either of the panel's own axes, so
  // its two inequalities carry no x or y: the tile is either cut by this slice
  // or it is not, and drawing its footprint anyway would be a lie.
  const cut = (depth) =>
    coveragePolygon({
      ...panel,
      slabs: slabsFor({
        centre: [0, 0, depth],
        u: [1, 0, 0],
        v: [0, 1, 0],
        matrix: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        lower: [-20, -20, -5],
        upper: [20, 20, 5],
      }),
    });

  assert.equal(cut(0).length, 4);
  assert.deepEqual(cut(45), []);
});

test("an axis the layer does not have constrains nothing", () => {
  // A 2D tile in a 3D space, or any layer without the panel's depth dimension:
  // it is present at every depth rather than at none. Such an axis produces no
  // slab at all, since there is no source dimension behind it.
  const polygon = coveragePolygon({
    ...panel,
    slabs: slabsFor({
      ...flat,
      matrix: identity2,
      lower: [-20, -20],
      upper: [20, 20],
    }),
  });

  assert.deepEqual(extent(polygon), [80, 30, 120, 70]);
});

test("an unbounded end of an axis is left unclipped", () => {
  const polygon = coveragePolygon({
    ...panel,
    slabs: slabsFor({
      ...flat,
      matrix: identity2,
      lower: [-20, -Infinity],
      upper: [Infinity, Infinity],
    }),
  });

  assert.deepEqual(extent(polygon), [80, 0, 200, 100]);
});

test("a cross-section oblique to every axis cuts the box as a hexagon", () => {
  // The panel's axes are whatever it says they are. A plane through the centre
  // of a cube with its normal along (1, 1, 1) meets six of the twelve edges,
  // and nothing here had to be told about the rotation to produce a hexagon.
  const width = 400;
  const half = 30;
  const u = [Math.SQRT1_2, -Math.SQRT1_2, 0];
  const v = [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)];

  const polygon = coveragePolygon({
    width,
    height: width,
    slabs: slabsFor({
      centre: [0, 0, 0],
      u,
      v,
      matrix: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      lower: [-half, -half, -half],
      upper: [half, half, half],
    }),
  });

  assert.equal(polygon.length, 6, "a plane cuts a cube in six edges");

  // Every vertex, taken back into the volume, has to land on the box.
  for (const [px, py] of polygon) {
    const x = px - width / 2;
    const y = py - width / 2;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = x * u[axis] + y * v[axis];
      assert.ok(
        Math.abs(value) <= half + 1e-6,
        `axis ${axis} left the box at ${value}`,
      );
    }
  }
});

test("a degenerate result is reported as nothing at all", () => {
  // A tile the slice grazes exactly is a line, and a line covers no pixels.
  const polygon = coveragePolygon({
    ...panel,
    slabs: slabsFor({
      ...flat,
      matrix: identity2,
      lower: [0, -20],
      upper: [0, 20],
    }),
  });

  assert.deepEqual(polygon, []);
  assert.deepEqual(coveragePolygon({ ...panel, width: 0, slabs: [] }), []);
});

test("clipping keeps the side the inequality names", () => {
  const square = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ];

  // x <= 0 halves it.
  assert.deepEqual(extent(clipToHalfPlane(square, 1, 0, 0)), [-1, -1, 0, 1]);
  // A plane parallel to the polygon keeps it whole or drops it whole, which is
  // what makes a depth axis fall out of the same code as every other.
  assert.deepEqual(clipToHalfPlane(square, 0, 0, 1), square);
  assert.deepEqual(clipToHalfPlane(square, 0, 0, -1), []);
});

test("a layer's own box is preferred to the box around it", () => {
  // A tile turned 30 degrees: the geometry that comes back is its own 40x60
  // grid and the map that places it, not the upright box it is inscribed in.
  const geometry = layerGeometry(
    placement({ leading: [2], linear: turn(Math.PI / 6), size: [40, 60] }),
    [1, 2],
  );

  assert.equal(geometry.bounds, undefined, "the bounding box is the fallback");
  assert.deepEqual(geometry.lower, [-0.5, -0.5]);
  assert.deepEqual(geometry.upper, [39.5, 59.5]);
});

test("a transform that mixes in an axis the viewer cannot state falls back", () => {
  // Contrived, but it is the guard that keeps the solve honest: if a source
  // dimension reached both a row being solved for and one that is not, its
  // coordinate could not be recovered from a position, and an outline drawn
  // anyway would be arbitrary. The upright box is used instead.
  const mixed = placement({ leading: [2], linear: turn(0), size: [40, 60] });
  // Let the channel axis contribute to a spatial row.
  mixed.matrix[(mixed.rank + 1) * 0 + 1] = 0.5;

  const geometry = layerGeometry(mixed, [1, 2]);
  assert.ok(geometry.bounds, "the mixed transform must not be solved");
  assert.equal(geometry.lower, undefined);
});

test("a rotation between axes of different spacing is read in metres", () => {
  // A Neuroglancer transform's linear coefficients act on *physical*
  // coordinates, so each one has to be scaled by the spacing of the source
  // dimension it consumes. Reading them with the output dimension's spacing
  // instead is invisible while the two axes are spaced alike - turning a tile
  // in a plane of square pixels - and shears the layer by their ratio the
  // moment they are not, which is every cross-section that cuts along z.
  const quarterTurn = [
    [0, -1],
    [1, 0],
  ];
  const geometry = layerGeometry(
    placement({ linear: quarterTurn, size: [10, 40], scales: [4, 1] }),
    [0, 1],
  );

  // Ten voxels along the coarse axis is forty metres, and a quarter turn puts
  // them on the other axis - still forty metres, not ten.
  const [x, y] = [0, 1].map(
    (row) =>
      geometry.matrix[row][0] * 10 +
      geometry.matrix[row][1] * 0 +
      geometry.translation[row],
  );
  assert.ok(Math.abs(x) < 1e-9, `x: ${x}`);
  assert.ok(Math.abs(y - 40) < 1e-9, `y: ${y}`);

  // And the same mistake, seen from the other side: a point well inside the
  // turned layer is on it.
  assert.equal(containsPoint(geometry, [-20, 20]), true);
  assert.equal(containsPoint(geometry, [20, 20]), false);
});

test("what counts as being on a layer is the layer, not the box around it", () => {
  // The corner a rotation swings *out* of the upright box, and the corner it
  // vacates. Judging by the box would get both of them wrong - and the outline
  // is drawn from this same geometry, so the two can only ever agree.
  const angle = Math.PI / 4;
  const geometry = layerGeometry(
    placement({ linear: turn(angle), size: [40, 40] }),
    [0, 1],
  );

  const source = (x, y) => {
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return [x * cos - y * sin, x * sin + y * cos];
  };

  // The middle of the tile, and a point just inside its far corner.
  assert.equal(containsPoint(geometry, source(20, 20)), true);
  assert.equal(containsPoint(geometry, source(39, 39)), true);

  // A corner of the upright box the turned tile does not reach.
  const upright = layerGeometry(
    placement({ linear: turn(0), size: [40, 40] }),
    [0, 1],
  );
  const corner = [-0.4, -0.4];
  assert.equal(containsPoint(upright, corner), true);
  assert.equal(
    containsPoint(geometry, corner),
    false,
    "a turned tile does not cover the corner of the box around it",
  );
});

test("a position past any bound is outside, on either shape", () => {
  const upright = layerGeometry(
    placement({ leading: [2], linear: turn(0), size: [64, 64] }),
    [1, 2],
  );

  assert.equal(containsPoint(upright, [32, 32]), true);
  assert.equal(containsPoint(upright, [32, 64]), false);
  assert.equal(containsPoint(upright, [-1, 32]), false);
  // Half-open: the upper face belongs to the next voxel along.
  assert.equal(containsPoint(upright, [63.5, 32]), false);

  const fallback = { bounds: [{ lower: 0, upper: 10 }] };
  assert.equal(containsPoint(fallback, [5]), true);
  assert.equal(containsPoint(fallback, [10]), false);
  assert.equal(containsPoint(fallback, [-0.001]), false);
});

test("bounds in the layer's own units are compared in them", () => {
  // The same physical position, in a layer whose voxels are twice the size.
  const coarse = layerGeometry(
    placement({ linear: turn(0), size: [64, 64], scales: [2, 2] }),
    [0, 1],
  );

  // 80 metres is 40 of the layer's own voxels, so still inside its 0..64.
  assert.equal(containsPoint(coarse, [80, 80]), true);
  // 200 is 100 of them, which is past its extent.
  assert.equal(containsPoint(coarse, [200, 80]), false);
});

test("solving is what turns a placement back into the layer's own axes", () => {
  const answer = solve(
    [
      [2, 1],
      [1, 3],
    ],
    [5, 10],
  );
  assert.ok(Math.abs(answer[0] - 1) < 1e-12, answer);
  assert.ok(Math.abs(answer[1] - 3) < 1e-12, answer);

  // Pivoting: a zero in the leading position is not a singular matrix.
  const pivoted = solve(
    [
      [0, 2],
      [3, 0],
    ],
    [4, 9],
  );
  assert.deepEqual(pivoted, [3, 2]);

  // A layer flattened along one of its own axes has no cross-section to draw.
  assert.equal(
    solve(
      [
        [1, 2],
        [2, 4],
      ],
      [1, 2],
    ),
    null,
  );
});
