/**
 * The app's only contact with Neuroglancer.
 *
 * Everything Neuroglancer-specific lives here: no other module imports it, so
 * upgrading the viewer means reading this file and nothing else. The rest of
 * the app talks in terms of layers, transforms, visibility and position.
 *
 * Only Neuroglancer's public API is used:
 *
 *   - `setupDefaultViewer({ target })` mounts a viewer into an element;
 *   - `viewer.state` is the documented, versioned viewer state - the same JSON
 *     that appears in a Neuroglancer link - with `toJSON`, `restoreState` and
 *     a `changed` signal;
 *   - `viewer.navigationState.position` holds the selected position;
 *   - `viewer.layerManager` owns the layers, each with a `setVisible`.
 *
 * Layers are described as viewer state rather than built object by object.
 * That is what the Python side already produces, it is the part of the API
 * with a stability guarantee, and it applies in place - so changing a
 * transform updates the running viewer instead of reloading it.
 */

import {
  setupDefaultViewer,
  StatusMessage,
} from "./neuroglancer/neuroglancer.js";

// This module is loaded as `viewer.js?v=<build>`; passing the same tag on
// means a rebuilt camera.js cannot be served from cache behind a fresh
// viewer.js - two halves of different builds fail in ways that look like
// neither.
const { carryCameraOver, centreOnData } = await import(
  `./camera.js${new URL(import.meta.url).search}`
);

//: Events that mean the user is driving the camera. Until one arrives, the
//: viewer keeps the camera on the data as the layers report their bounds.
const TAKEOVER_EVENTS = ["pointerdown", "wheel", "keydown"];

/** Wraps one mounted Neuroglancer instance. */
export class NeuroglancerViewer {
  #viewer = null;
  #target = null;
  #stateListeners = new Set();
  #positionListeners = new Set();
  #onStateChanged = null;
  #onPositionChanged = null;
  #onCoordinateSpaceChanged = null;
  #applying = false;
  #placing = false;
  #cameraPlaced = false;
  #onTakeover = null;
  #lastNames = null;
  #lastPosition = null;

  /**
   * Mount a viewer into `target`.
   *
   * Safe to call once; call `dispose()` before mounting another.
   */
  mount(target) {
    if (this.#viewer) throw new Error("the viewer is already mounted");

    this.#target = target;
    this.#viewer = setupDefaultViewer({ target });

    // Fresh viewer, so the camera is ours to place until the user moves it.
    this.#cameraPlaced = false;
    this.#onTakeover = () => {
      if (!this.#placing) this.#cameraPlaced = true;
    };
    for (const type of TAKEOVER_EVENTS) {
      target.addEventListener(type, this.#onTakeover, {
        capture: true,
        passive: true,
      });
    }

    // Neuroglancer coalesces its own updates, so these fire once per settled
    // change rather than once per mutation.
    this.#onStateChanged = () => {
      // Ignore the echo of our own write: listeners want user-driven changes.
      if (this.#applying) return;
      const state = this.getState();
      for (const listener of this.#stateListeners) listener(state);
    };
    this.#viewer.state.changed.add(this.#onStateChanged);

    this.#onPositionChanged = () => {
      this.#noteUserPosition();
      const position = this.getPosition();
      for (const listener of this.#positionListeners) listener(position);
    };
    this.#viewer.navigationState.position.changed.add(
      this.#onPositionChanged,
    );

    // Rebuilding the global coordinate space is the moment a carried-over
    // camera can stop meaning what it did. See `#syncCameraToSpace`.
    this.#onCoordinateSpaceChanged = () => this.#syncCameraToSpace();
    this.#viewer.navigationState.coordinateSpace.changed.add(
      this.#onCoordinateSpaceChanged,
    );

    return this;
  }

  /**
   * Keep the camera pointing at the same place when the axes are rebuilt.
   *
   * Neuroglancer keeps the camera across a state change and, when the global
   * coordinate space is rebuilt, remaps the position *by dimension index*.
   * That space is assembled from the layers as they load, so its dimension
   * order is not the order the state asked for: applying a state can centre
   * the camera while the axes are `t, z, y, x` and then have the space settle
   * into `x, y, z, t`, at which point every coordinate means a different axis.
   * The camera ends up somewhere the data is not - every layer loaded, nothing
   * rendered - which is what made switching back to a transform key show an
   * empty viewer.
   *
   * So the position is carried over by dimension *name* instead, which is what
   * the index remap was standing in for. An axis that is genuinely new, or one
   * that still lands outside the data, falls back to the middle of the volume.
   */
  #syncCameraToSpace() {
    const navigation = this.#viewer?.navigationState;
    const space = navigation?.coordinateSpace.value;
    if (!space || !space.valid) return;

    const position = navigation.position.value;
    if (!position || position.length !== space.rank) return;

    const names = Array.from(space.names);
    const geometry = {
      names,
      position: Array.from(position),
      lowerBounds: Array.from(space.bounds.lowerBounds),
      upperBounds: Array.from(space.bounds.upperBounds),
    };

    // Until the user takes over, keep the camera on the middle of whatever
    // has loaded; Neuroglancer's own placement happens on the first valid
    // coordinate space, which can be before every layer has reported.
    const next = this.#cameraPlaced
      ? carryCameraOver({
          ...geometry,
          previousNames: this.#lastNames,
          previousPosition: this.#lastPosition,
        })
      : centreOnData(geometry);

    if (next) {
      this.#placing = true;
      try {
        navigation.position.value = Float32Array.from(next);
      } finally {
        this.#placing = false;
      }
    }
    this.#rememberCamera(names, next ?? position);
  }

  #rememberCamera(names, position) {
    this.#lastNames = names;
    this.#lastPosition = Array.from(position);
  }

  /**
   * Record a move the user made, so the next rebuild can carry it over.
   *
   * Skipped while the axes are mid-change: the position reported then is the
   * index-remapped one this class exists to correct, and recording it would
   * make that wrong value the reference.
   */
  #noteUserPosition() {
    const navigation = this.#viewer?.navigationState;
    const space = navigation?.coordinateSpace.value;
    if (!space || !space.valid || this.#lastNames === null) return;

    const names = space.names;
    if (
      names.length !== this.#lastNames.length ||
      this.#lastNames.some((name, i) => name !== names[i])
    ) {
      return;
    }

    const position = navigation.position.value;
    if (position && position.length === space.rank) {
      this.#lastPosition = Array.from(position);
    }
  }

  get mounted() {
    return this.#viewer !== null;
  }

  /**
   * Put the viewer back to how it was when it was mounted.
   *
   * Emptying the layer list is not enough: the dimensions, their scales and
   * units, the camera and the layout all outlive it, and the next dataset is
   * then shown through the previous one's coordinate space. The remembered
   * camera goes too - carrying it into an unrelated dataset is how a fresh
   * load ends up pointing at nothing.
   */
  reset() {
    const target = this.#require() && this.#target;

    // `setupDefaultViewer` binds the viewer state to the URL fragment, and a
    // new instance restores from it on the way up - so without this the
    // discarded dataset comes straight back. Replaced rather than assigned,
    // to avoid a history entry and a hashchange.
    if (globalThis.location?.hash) {
      globalThis.history?.replaceState(
        null,
        "",
        globalThis.location.pathname + globalThis.location.search,
      );
    }

    // Rebuilt rather than emptied. `state.reset()` leaves the combined
    // coordinate space behind, so the next dataset is measured in the
    // previous one's dimensions and bounds - which is how a load after a
    // clear ended up showing nothing. Listeners registered through
    // `onStateChanged`/`onPositionChanged` are kept and re-wired.
    this.dispose();
    this.mount(target);
  }

  /** Tear the viewer down and release its WebGL context and workers. */
  dispose() {
    if (!this.#viewer) return;

    this.#viewer.state.changed.remove(this.#onStateChanged);
    this.#viewer.navigationState.position.changed.remove(
      this.#onPositionChanged,
    );
    this.#viewer.navigationState.coordinateSpace.changed.remove(
      this.#onCoordinateSpaceChanged,
    );
    if (this.#onTakeover) {
      for (const type of TAKEOVER_EVENTS) {
        this.#target?.removeEventListener(type, this.#onTakeover, {
          capture: true,
        });
      }
    }
    this.#viewer.dispose();

    this.#viewer = null;
    this.#onTakeover = null;
    this.#onStateChanged = null;
    this.#onPositionChanged = null;
    this.#onCoordinateSpaceChanged = null;
    this.#lastNames = null;
    this.#lastPosition = null;
    if (this.#target) this.#target.replaceChildren();
    this.#target = null;
  }

  #require() {
    if (!this.#viewer) throw new Error("the viewer is not mounted");
    return this.#viewer;
  }

  // -------------------------------------------------------------------
  // Viewer state
  // -------------------------------------------------------------------

  /** The current viewer state, as the JSON a Neuroglancer link carries. */
  getState() {
    return this.#require().state.toJSON();
  }

  /**
   * Apply a viewer state, in place.
   *
   * This is what keeps a transform change immediate: the running viewer is
   * updated, so nothing is reloaded and the WebGL context, the chunk workers
   * and anything already fetched all survive.
   *
   * Neuroglancer's `restoreState` only touches the keys it is given, so a
   * state that describes layers but not the camera leaves the camera alone -
   * which is exactly what is wanted when only a transform changed. Resetting
   * first would send the camera back to the origin, and nothing moves it back
   * unless a *new* data source loads: switching between two transform keys
   * reuses the same source URLs, so the view would simply go blank.
   *
   * Pass `preserveView: false` to let the incoming state define the view,
   * dropping whatever the user had navigated to.
   */
  setState(state, { preserveView = true } = {}) {
    const viewer = this.#require();

    this.#applying = true;
    try {
      if (!preserveView) viewer.state.reset();
      viewer.state.restoreState(state);
    } finally {
      this.#applying = false;
    }
  }

  // -------------------------------------------------------------------
  // Layers
  // -------------------------------------------------------------------

  /** Names of the current layers, in order. */
  getLayerNames() {
    return this.#require().layerManager.managedLayers.map(
      (layer) => layer.name,
    );
  }

  /**
   * Replace the layers, leaving the rest of the viewer state alone.
   *
   * Each layer is a Neuroglancer layer specification: `{name, type, source,
   * ...}`, where `source` may carry a `transform` (see `withTransform`).
   */
  setLayers(layers) {
    const state = this.getState();
    this.setState({ ...state, layers });
  }

  /**
   * Re-aim each named layer's source, changing nothing else.
   *
   * This is how a transform_key switch is applied. Going through the viewer
   * state instead would mean handing Neuroglancer a `layers` array, and
   * restoring that clears the layer list and builds every layer again: the
   * shader and its contrast range, the selected layer, the chosen layout and
   * anything else the user had set are all constructed anew. Rebuilt image
   * layers also come up before their contrast range has been computed, so
   * they render black until something makes them re-read it.
   *
   * A loaded source's transform is separately watchable - it is what the
   * layer's own "Source" tab edits - so assigning it moves the data and
   * leaves the rest of the layer alone.
   *
   * `transforms` maps layer name to a `{matrix, outputDimensions}` spec, or
   * to null for no transform.
   */
  setLayerTransforms(transforms) {
    const viewer = this.#require();
    const missing = [];

    for (const [url, transform] of Object.entries(transforms)) {
      const sources = this.#sourcesReading(url);
      if (!sources.length) {
        missing.push(url);
        continue;
      }
      for (const dataSource of sources) applyTransform(dataSource, transform);
    }

    if (missing.length) {
      throw new Error(`no layer reads from: ${missing.join(", ")}`);
    }
  }

  /**
   * The loaded data sources reading from `url`.
   *
   * Layers are addressed by the URL they read rather than by name: the name
   * is Neuroglancer's to change. Opening an OME-Zarr that carries omero
   * metadata renames the layer after its channel - `"0: tile.ome.zarr"`
   * becomes `"0: tile.ome.zarr channel 0"` - and a name can also be made
   * unique, or edited by the user. The URL is the app's own handle on a
   * layer and survives all of that.
   *
   * One URL can back several layers, since a multi-channel image is opened
   * as one layer per channel, so every match is returned.
   */
  #sourcesReading(url) {
    const sources = [];
    for (const managed of this.#viewer.layerManager.managedLayers) {
      for (const dataSource of managed.layer?.dataSources ?? []) {
        if (dataSource.spec?.url === url) sources.push(dataSource);
      }
    }
    return sources;
  }

  /**
   * Update named layers in place, merging each patch into that layer.
   *
   * Layers not named are left untouched. Note that this still goes through
   * the viewer state, so the layers are rebuilt; prefer
   * `setLayerTransforms` or `setLayerVisibility` when they will do.
   */
  updateLayers(patches) {
    const state = this.getState();
    const byName = new Map(Object.entries(patches));

    const layers = (state.layers || []).map((layer) => {
      const patch = byName.get(layer.name);
      if (!patch) return layer;
      byName.delete(layer.name);
      return mergeLayer(layer, patch);
    });

    if (byName.size) {
      throw new Error(
        `no such layer(s): ${[...byName.keys()].join(", ")}`,
      );
    }

    this.setState({ ...state, layers });
  }

  /**
   * Show or hide layers, addressed by the URL they read.
   *
   * Goes through `ManagedUserLayer.setVisible`, so it takes effect without
   * touching the rest of the state. See `#sourcesReading` for why the URL is
   * the handle rather than the layer name.
   */
  setLayerVisibility(visibility) {
    const viewer = this.#require();
    const missing = [];

    for (const [url, visible] of Object.entries(visibility)) {
      let found = false;
      for (const managed of viewer.layerManager.managedLayers) {
        const reads = (managed.layer?.dataSources ?? []).some(
          (dataSource) => dataSource.spec?.url === url,
        );
        if (!reads) continue;
        managed.setVisible(Boolean(visible));
        found = true;
      }
      if (!found) missing.push(url);
    }

    if (missing.length) {
      throw new Error(`no layer reads from: ${missing.join(", ")}`);
    }
  }

  // -------------------------------------------------------------------
  // Position
  // -------------------------------------------------------------------

  /** The selected position, in the viewer's output dimensions. */
  getPosition() {
    const position = this.#require().navigationState.position;
    return Array.from(position.value ?? []);
  }

  /** Move the selected position. */
  setPosition(position) {
    const navigation = this.#require().navigationState.position;
    navigation.value = Float32Array.from(position);
  }

  /** The dimension names the position is expressed in. */
  getPositionDimensions() {
    const space = this.#require().navigationState.coordinateSpace.value;
    return space ? Array.from(space.names) : [];
  }

  // -------------------------------------------------------------------
  // Callbacks
  // -------------------------------------------------------------------

  /** Call `listener(state)` when the user changes the viewer state. */
  onStateChanged(listener) {
    this.#stateListeners.add(listener);
    return () => this.#stateListeners.delete(listener);
  }

  /** Call `listener(position)` when the selected position moves. */
  onPositionChanged(listener) {
    this.#positionListeners.add(listener);
    return () => this.#positionListeners.delete(listener);
  }

  /** Show a message in the viewer's own status area. */
  showStatus(message, { timeoutMs = 5000 } = {}) {
    StatusMessage.showTemporaryMessage(String(message), timeoutMs);
  }
}

/**
 * Attach an affine transform to a layer source.
 *
 * Neuroglancer expects the matrix without its final row, and its translation
 * column in output units. `outputDimensions` names and scales those units.
 */
export function withTransform(source, matrix, outputDimensions) {
  const url = typeof source === "string" ? source : source.url;
  const rest = typeof source === "string" ? {} : { ...source };
  delete rest.url;

  if (!matrix) return { ...rest, url };

  return {
    ...rest,
    url,
    transform: {
      matrix: matrix.slice(0, -1).map((row) => Array.from(row, Number)),
      ...(outputDimensions ? { outputDimensions } : {}),
    },
  };
}

/**
 * Point one data source at a new transform, once it is in a position to take
 * it.
 *
 * A source that is still loading has no transform to assign yet, so the
 * change is made when it arrives - a transform_key switched during the first
 * load would otherwise be silently dropped.
 */
function applyTransform(dataSource, transform) {
  const apply = () => {
    const loadState = dataSource.loadState;
    if (!loadState || loadState.error) return false;
    // `restoreState(undefined)` resets to the source's own transform, which
    // is what "no transform" has to mean here.
    loadState.transform.restoreState(transform ?? undefined);
    return true;
  };

  if (apply()) return;

  const stop = dataSource.changed.add(() => {
    if (apply()) stop();
  });
}

/** Merge a patch into a layer specification, combining `source` sensibly. */
function mergeLayer(layer, patch) {
  const merged = { ...layer, ...patch };

  if (patch.source && layer.source && !Array.isArray(patch.source)) {
    const base =
      typeof layer.source === "string" ? { url: layer.source } : layer.source;
    const next =
      typeof patch.source === "string" ? { url: patch.source } : patch.source;
    merged.source = { ...base, ...next };
  }

  return merged;
}
