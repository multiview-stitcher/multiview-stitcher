/**
 * Where a layer lands on a cross-section panel.
 *
 * Pure 2D geometry and linear algebra, deliberately free of any Neuroglancer or
 * DOM import, so it can be exercised under `node --test`. `viewer.js` owns
 * reading the panels and drawing the result; what is decided here is the
 * *shape* a highlight takes.
 *
 * The problem looks three-dimensional and is not. A panel shows the plane
 *
 *     P(x, y) = centre + x·u + y·v
 *
 * where `u` and `v` are the physical vectors one viewport pixel spans, across
 * and down. A layer occupies a box in its *own* source coordinates - the voxel
 * grid - and its transform carries that box into the viewer's space, where a
 * rotated tile is a parallelepiped rather than an axis-aligned box.
 *
 * Pulling the panel's plane back through that transform turns the whole thing
 * into two dimensions: each source coordinate becomes an affine function of the
 * pixel position, `at + x·perX + y·perY`, and the layer's extent along that
 * source axis becomes a pair of parallel lines on the panel. The region covered
 * is the intersection of those slabs, which is a convex polygon.
 *
 * Three things fall out of that framing rather than having to be handled:
 *
 *   - a *rotated tile* is outlined at its own angle, because the slabs are its
 *     own axes rather than the viewer's;
 *   - a cross-section the user has rotated is no different: `u` and `v` are
 *     whatever the panel says they are, and the polygon comes out as the
 *     hexagon a tilted plane really cuts from a box;
 *   - a slice that misses the layer gives an empty polygon. Its depth axis then
 *     has no component along `u` or `v`, so that slab's two inequalities carry
 *     no `x` or `y`: they are either both true, and the panel cuts the layer,
 *     or both false, and nothing is drawn.
 */

/**
 * Where a layer is, in terms the viewer's own coordinate space can state.
 *
 * `placement` is the layer's transform as plain data - `viewer.js` reads it off
 * Neuroglancer, and nothing Neuroglancer-shaped reaches this module:
 *
 *   - `rank`, `sourceRank`: the output and source dimension counts;
 *   - `matrix`: `(rank + 1) x (sourceRank + 1)`, column-major, mapping source
 *     indices to output indices - the layer's placement, rotation and all;
 *   - `sourceLower`, `sourceUpper`: the layer's own box, in source indices;
 *   - `outputLower`, `outputUpper`: the axis-aligned box Neuroglancer reports
 *     around the *result*, in output indices;
 *   - `outputScales`: output indices to metres.
 *
 * `rows` names the output dimensions to work in: the ones the viewer's own
 * space also has, so that a position can be stated in them. Every array that
 * later goes in or comes out is indexed the same way.
 *
 * Two shapes come back, and which one says how faithfully the layer can be
 * described. Preferred is the layer's own box with the map that places it,
 * which is the *image*, edge for edge, whatever rotation it carries. The
 * fallback is the axis-aligned box, used when the transform mixes these
 * dimensions with ones the viewer cannot state a position in - an
 * over-covering answer beats none.
 */
export function layerGeometry(placement, rows) {
  return (
    sourceGeometry(placement, rows) ?? {
      bounds: rows.map((row) => ({
        lower: placement.outputLower[row] * placement.outputScales[row],
        upper: placement.outputUpper[row] * placement.outputScales[row],
      })),
    }
  );
}

/**
 * The layer's own box and the map that places it, or null.
 *
 * Only the source dimensions that map *purely* into `rows` are taken, and only
 * if those rows are reached from no others: what is left is then a square
 * system between the two, which is what makes the map invertible back into
 * source coordinates. A layer-local axis such as a channel drops out here,
 * which is right - a tile covers every channel and constrains nothing along
 * one.
 */
function sourceGeometry({ rank, sourceRank, matrix, sourceLower, sourceUpper, outputScales }, rows) {
  if (!rows.length) return null;
  const at = (row, column) => matrix[(rank + 1) * column + row];

  const known = new Set(rows);
  const columns = [];
  for (let column = 0; column < sourceRank; column += 1) {
    let reachesKnown = false;
    let reachesOther = false;
    for (let row = 0; row < rank; row += 1) {
      if (at(row, column) === 0) continue;
      if (known.has(row)) reachesKnown = true;
      else reachesOther = true;
    }
    if (reachesKnown && reachesOther) return null;
    if (reachesKnown) columns.push(column);
  }

  if (columns.length !== rows.length) return null;

  // Nothing outside those columns may reach a known row either, or the mapping
  // would depend on a source coordinate that is not being solved for.
  for (const row of known) {
    for (let column = 0; column < sourceRank; column += 1) {
      if (columns.includes(column)) continue;
      if (at(row, column) !== 0) return null;
    }
  }

  return {
    matrix: rows.map((row) =>
      columns.map((column) => at(row, column) * outputScales[row]),
    ),
    translation: rows.map((row) => at(row, sourceRank) * outputScales[row]),
    lower: columns.map((column) => sourceLower[column]),
    upper: columns.map((column) => sourceUpper[column]),
  };
}

/**
 * The layer's own coordinates for a point given in the viewer's, or null.
 *
 * One linear solve. Everything the outline and the hit test would otherwise
 * disagree about - which corner of a turned tile counts as being on it - is
 * settled here, once, for both.
 */
function sourceCoordinates(geometry, point) {
  return solve(
    geometry.matrix,
    point.map((value, i) => value - geometry.translation[i]),
  );
}

/**
 * Whether a point, in the viewer's coordinates, is on the layer.
 *
 * Half-open, as a voxel grid is: a point on the upper face belongs to the next
 * layer along, not this one. A non-finite bound places no constraint.
 */
export function containsPoint(geometry, point) {
  if (geometry.bounds) {
    return geometry.bounds.every(
      (bound, i) => !(point[i] < bound.lower) && !(point[i] >= bound.upper),
    );
  }

  const source = sourceCoordinates(geometry, point);
  return (
    source !== null &&
    source.every(
      (value, i) =>
        !(value < geometry.lower[i]) && !(value >= geometry.upper[i]),
    )
  );
}

/**
 * One slab per axis of the layer, as one panel's plane cuts it.
 *
 * The panel shows `centre + x·u + y·v`, all three stated in the viewer's
 * coordinates. Solving that back through the layer's own map gives each source
 * coordinate as an affine function of the pixel position - which is the whole
 * of what `coveragePolygon` needs, and is where a rotated tile stops being a
 * special case.
 */
export function layerSlabs(geometry, centre, u, v) {
  if (geometry.bounds) {
    return geometry.bounds.map((bound, i) => ({
      at: centre[i],
      perX: u[i],
      perY: v[i],
      lower: bound.lower,
      upper: bound.upper,
    }));
  }

  const at = sourceCoordinates(geometry, centre);
  const perX = solve(geometry.matrix, u);
  const perY = solve(geometry.matrix, v);
  if (!at || !perX || !perY) return null;

  return at.map((value, i) => ({
    at: value,
    perX: perX[i],
    perY: perY[i],
    lower: geometry.lower[i],
    upper: geometry.upper[i],
  }));
}

/**
 * Solve `matrix · answer = rhs` for a small dense square system.
 *
 * Gaussian elimination with partial pivoting. Returns null for a singular
 * matrix, which is a layer flattened to nothing along one of its own axes -
 * there is no cross-section to draw, and no answer to give.
 *
 * `matrix` is an array of rows. It is not modified.
 */
export function solve(matrix, rhs) {
  const n = rhs.length;
  const rows = matrix.map((row, i) => [...row, rhs[i]]);

  for (let column = 0; column < n; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < n; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) {
        pivot = row;
      }
    }
    if (!(Math.abs(rows[pivot][column]) > 0)) return null;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];

    for (let row = 0; row < n; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column] / rows[column][column];
      if (factor === 0) continue;
      for (let k = column; k <= n; k += 1) {
        rows[row][k] -= factor * rows[column][k];
      }
    }
  }

  // Eliminated above and below the pivot, so what is left is diagonal.
  return rows.map((row, i) => row[n] / row[i]);
}

/**
 * The part of `polygon` on the near side of `a·x + b·y = c`.
 *
 * Sutherland-Hodgman against a single edge: walk the ring, keep the vertices
 * that satisfy the inequality, and add the crossing point wherever an edge
 * changes side. A convex polygon stays convex, so the six calls that make up a
 * box can simply be chained.
 *
 * `a` and `b` are both zero when the plane is parallel to the face - the
 * ordinary case of a depth axis in an axis-aligned view. Every vertex then has
 * the same signed distance, so the polygon is kept whole or dropped whole,
 * which is exactly right.
 */
export function clipToHalfPlane(polygon, a, b, c) {
  const clipped = [];
  const distance = ([x, y]) => a * x + b * y - c;

  for (let i = 0; i < polygon.length; i += 1) {
    const current = polygon[i];
    const next = polygon[(i + 1) % polygon.length];
    const here = distance(current);
    const there = distance(next);

    if (here <= 0) clipped.push(current);
    if (here <= 0 !== there <= 0) {
      // The two distances cannot be equal here: they are on opposite sides.
      const t = here / (here - there);
      clipped.push([
        current[0] + t * (next[0] - current[0]),
        current[1] + t * (next[1] - current[1]),
      ]);
    }
  }

  return clipped;
}

/**
 * The region of one panel a layer covers, in panel pixels.
 *
 * Each slab describes one axis of the layer as seen on this panel: `at` is the
 * coordinate along it at the centre of the panel, `perX` and `perY` how fast it
 * changes per pixel across and down, and `lower`/`upper` the extent the layer
 * occupies. A non-finite end is an axis the layer does not bound.
 *
 * The result starts from the panel rectangle, so it is already clipped to what
 * is on screen. Returns the polygon's vertices with (0, 0) at the panel's
 * top-left corner, or an empty array when the panel shows nothing of the layer.
 */
export function coveragePolygon({ width, height, slabs }) {
  if (!(width > 0) || !(height > 0)) return [];

  let polygon = [
    [-width / 2, -height / 2],
    [width / 2, -height / 2],
    [width / 2, height / 2],
    [-width / 2, height / 2],
  ];

  for (const slab of slabs) {
    if (!polygon.length) break;
    const { at, perX, perY, lower, upper } = slab;

    if (Number.isFinite(upper)) {
      polygon = clipToHalfPlane(polygon, perX, perY, upper - at);
    }
    if (polygon.length && Number.isFinite(lower)) {
      polygon = clipToHalfPlane(polygon, -perX, -perY, at - lower);
    }
  }

  // A polygon that survived as a point or a line covers no pixels. That is not
  // a rounding artefact to tidy away: it is a slice grazing the very edge of a
  // layer, or a layer with no extent along one of the panel's own axes, and
  // drawing it would put a stray line across the data.
  if (polygon.length < 3 || area(polygon) < MINIMUM_AREA_PX) return [];

  return polygon.map(([x, y]) => [x + width / 2, y + height / 2]);
}

//: Square pixels below which an outline is not worth drawing.
const MINIMUM_AREA_PX = 1;

/** The area of a polygon: the shoelace sum, as a magnitude. */
function area(polygon) {
  let sum = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const [x1, y1] = polygon[i];
    const [x2, y2] = polygon[(i + 1) % polygon.length];
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}
