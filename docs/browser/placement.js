/**
 * The rules behind manual tile placement.
 *
 * Pure array logic, deliberately free of any Neuroglancer import, so it can be
 * exercised under `node --test` without a browser. `viewer.js` owns the pointer
 * handling and every call into Neuroglancer; what is decided here is *which*
 * tile a drag moves and *how far*, which is where the dimension bookkeeping -
 * and so the bugs - live.
 *
 * Three coordinate systems meet in this file:
 *
 *   - *display* coordinates: the three axes on screen, in the units of the
 *     global coordinate space dimensions they are drawn from;
 *   - *global* coordinates: the viewer's own space, one value per dimension of
 *     `navigationState.coordinateSpace`;
 *   - a layer's *output* dimensions: the space its source transform maps into,
 *     which is where a translation has to end up. Its dimensions carry their
 *     own scales, and a layer may not have all of the global ones.
 *
 * Dimensions are matched by *name* throughout. Matching by index is what the
 * viewer's camera code already had to stop doing: the global space is assembled
 * from the layers as they load, so its order is not the order anything else
 * uses.
 */

/**
 * Which layer a drag should move.
 *
 * The pointer decides when it can: one tile under it is unambiguous. Where
 * tiles overlap - which, for a tiled acquisition, is most of the interesting
 * places - the pointer cannot, so the choice falls to the tile the user
 * selected in the views list. With nothing under the pointer there is nothing
 * to move, and an overlap with no selection is left alone rather than guessed
 * at: moving the wrong tile is worse than moving none.
 *
 * Returns `{url, reason}`, where `url` is null when no drag should start.
 */
export function pickDragTarget(candidates, selected) {
  const urls = Array.from(new Set(candidates));

  if (urls.length === 0) return { url: null, reason: "empty" };
  if (urls.length === 1) return { url: urls[0], reason: "only" };
  if (selected && urls.includes(selected)) {
    return { url: selected, reason: "selected" };
  }
  return { url: null, reason: "ambiguous" };
}

/**
 * Whether a global position falls inside one layer's transformed bounds.
 *
 * The bounds come from the layer's own output space, so they already carry
 * whatever transform the layer is being shown under. Dimensions the layer does
 * not have - a channel dimension is local to the layer, and the global space
 * has no `c'` - place no constraint: a tile is under the pointer whatever
 * channel is on screen.
 */
export function boundsContain(
  { names, scales, lowerBounds, upperBounds },
  position,
  globalNames,
  globalScales,
) {
  for (let output = 0; output < names.length; output += 1) {
    const global = globalNames.indexOf(names[output]);
    if (global === -1) continue;

    const lower = lowerBounds[output];
    const upper = upperBounds[output];
    if (!Number.isFinite(lower) && !Number.isFinite(upper)) continue;

    const value =
      position[global] * (globalScales[global] / scales[output]);
    if (value < lower || value >= upper) return false;
  }
  return true;
}

/**
 * One layer's translation, in its own output dimensions, for a drag.
 *
 * `displayDelta` is how far the drag moved in display coordinates - which is
 * what a panel reports, and is already in the plane of that panel, so a drag in
 * a cross-section never moves the tile through the slice. Dimensions the layer
 * does not have are dropped, and the rest are rescaled from global units into
 * the layer's own.
 */
export function translationForDrag({
  displayDelta,
  displayDimensionIndices,
  globalNames,
  globalScales,
  outputNames,
  outputScales,
}) {
  const translation = new Array(outputNames.length).fill(0);

  for (let display = 0; display < displayDelta.length; display += 1) {
    const global = displayDimensionIndices[display];
    if (global === undefined || global < 0) continue;

    const output = outputNames.indexOf(globalNames[global]);
    if (output === -1) continue;

    translation[output] +=
      displayDelta[display] * (globalScales[global] / outputScales[output]);
  }

  return translation;
}

/**
 * A copy of a Neuroglancer transform matrix, moved by `translation`.
 *
 * Neuroglancer stores the matrix column-major with a stride of `rank + 1`, so
 * the translation is its last column. Only that column is touched: whatever
 * rotation or scale the layer already carries is the coordinate system the drag
 * happens in, not something a drag may change.
 */
export function translateMatrix(matrix, rank, translation) {
  const moved = Float64Array.from(matrix);
  const stride = rank + 1;

  for (let i = 0; i < translation.length && i < rank; i += 1) {
    moved[stride * rank + i] += translation[i];
  }

  return moved;
}

// ---------------------------------------------------------------------------
// Rotation
// ---------------------------------------------------------------------------
//
// A rotation drag turns a tile about its own centre, in the plane of the panel
// it is dragged in. Two things make that more than a 2x2 matrix:
//
//   - the plane is the panel's, not a coordinate plane. Neuroglancer lets the
//     user rotate a cross-section, and the rotation has to stay in whatever
//     plane is on screen;
//   - display coordinates are voxels, and voxels are not cubes. A rotation
//     applied to anisotropic voxel counts is a shear, so the arithmetic below
//     happens in physical units and is converted back per dimension.

/** Component-wise product: a display-space vector in physical units. */
export function toPhysical(vector, scales) {
  return Array.from(vector, (value, i) => value * (scales[i] ?? 1));
}

/**
 * Where a physical offset lands, in viewport pixels.
 *
 * `u` and `v` are the physical vectors one viewport pixel spans, across and
 * down. The panel's projection is a similarity - rotation and one scale - so a
 * projection onto each is all that is needed, and no matrix inverse.
 */
export function pixelOffset(offset, u, v) {
  const project = (basis) => {
    const square = dot(basis, basis);
    return square === 0 ? 0 : dot(offset, basis) / square;
  };
  return { x: project(u), y: project(v) };
}

/** How far a drag has swept around a centre, in radians. */
export function dragAngle(centre, from, to) {
  return (
    Math.atan2(to.y - centre.y, to.x - centre.x) -
    Math.atan2(from.y - centre.y, from.x - centre.x)
  );
}

/**
 * Rotation by `angle` within the plane spanned by `u` and `v`.
 *
 * Built in the panel's own frame rather than about a coordinate axis, so a
 * cross-section the user has rotated still turns tiles in the plane they see.
 * Anything along the plane's normal is left where it is, which is what makes
 * this the 2D rotation the drag looks like.
 */
export function planeRotation(u, v, angle) {
  const uHat = normalize(u);
  const vHat = normalize(v);
  const nHat = normalize(cross(uHat, vHat));
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  // n n^T + cos (u u^T + v v^T) + sin (v u^T - u v^T)
  return Array.from({ length: 3 }, (_, i) =>
    Array.from(
      { length: 3 },
      (_, j) =>
        nHat[i] * nHat[j] +
        cos * (uHat[i] * uHat[j] + vHat[i] * vHat[j]) +
        sin * (vHat[i] * uHat[j] - uHat[i] * vHat[j]),
    ),
  );
}

// A Neuroglancer source transform is not in one set of units. Its *linear*
// coefficients act on physical coordinates - Neuroglancer rescales them by the
// input and output dimension scales itself - while its translation column is
// in output pixels. That mixture is why `_affine_to_neuroglancer_source_
// transform` on the Python side divides only the translation by the spacing.
//
// A rotation is therefore built and composed in a matrix that is physical
// throughout, and converted back at the end. Two consequences of getting this
// wrong were invisible in an xy view, where the two axes share a spacing, and
// obvious in xz or zy, where a z step is four y steps: the linear block came
// out sheared, and composing it onto the transform the tile already had moved
// the centre it was supposed to turn about.

/** A Neuroglancer transform matrix with its translation in physical units. */
export function toPhysicalMatrix(matrix, rank, outputScales) {
  const physical = Float64Array.from(matrix);
  const stride = rank + 1;
  for (let row = 0; row < rank; row += 1) {
    physical[stride * rank + row] *= outputScales[row];
  }
  return physical;
}

/** The inverse of `toPhysicalMatrix`: back into Neuroglancer's units. */
export function fromPhysicalMatrix(matrix, rank, outputScales) {
  const converted = Float64Array.from(matrix);
  const stride = rank + 1;
  for (let row = 0; row < rank; row += 1) {
    converted[stride * rank + row] /= outputScales[row];
  }
  return converted;
}

/**
 * The transform that turns one layer about `centre`, in physical units.
 *
 * `axes` gives the output dimension each of the three display axes is drawn
 * from, or -1 where the layer has none - so a rotation in a plane the layer
 * does not span simply leaves it alone. `centre` is in the layer's own output
 * coordinates, one value per output dimension, and is converted here.
 *
 * The linear block is the rotation itself, with no reference to the dimension
 * scales: it is already in the units Neuroglancer reads it in. Returned in
 * physical form, to be applied *after* the layer's own transform - see
 * `composeAffine` and `toPhysicalMatrix`.
 */
export function rotationMatrix({ rotation, centre, axes, outputScales, rank }) {
  const stride = rank + 1;
  const affine = new Float64Array(stride * stride);
  for (let i = 0; i < stride; i += 1) affine[stride * i + i] = 1;

  for (let a = 0; a < 3; a += 1) {
    const row = axes[a];
    if (row === undefined || row < 0) continue;

    // Turning about a point means translating by however far that point moves
    // back: c - Rc, in the physical units the rotation is expressed in.
    let translation = centre[row] * outputScales[row];

    for (let b = 0; b < 3; b += 1) {
      const column = axes[b];
      if (column === undefined || column < 0) continue;

      affine[stride * column + row] = rotation[a][b];
      translation -= rotation[a][b] * centre[column] * outputScales[column];
    }

    affine[stride * rank + row] = translation;
  }

  return affine;
}

/**
 * `a` applied after `b`, for two homogeneous matrices in physical units.
 *
 * Column-major with a stride of `rank + 1`, so this is an ordinary matrix
 * product that carries the translations along with it. Both matrices must be
 * physical throughout: a translation column in output pixels does not survive
 * being multiplied by another matrix's linear block.
 */
export function composeAffine(a, b, rank) {
  const stride = rank + 1;
  const out = new Float64Array(stride * stride);

  for (let column = 0; column < stride; column += 1) {
    for (let row = 0; row < stride; row += 1) {
      let sum = 0;
      for (let k = 0; k < stride; k += 1) {
        sum += a[stride * k + row] * b[stride * column + k];
      }
      out[stride * column + row] = sum;
    }
  }

  return out;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * (b[i] ?? 0);
  return sum;
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize(vector) {
  const length = Math.sqrt(dot(vector, vector));
  return length === 0 ? [0, 0, 0] : Array.from(vector, (v) => v / length);
}
