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
 *   - `viewer.layerManager` owns the layers, each with a `setVisible`;
 *   - `viewer.toolPalettes` and the side panel states are the same objects the
 *     `toolPalettes`, `layerListPanel` and `selectedLayer` state keys address.
 *
 * Layers are described as viewer state rather than built object by object.
 * That is what the Python side already produces, it is the part of the API
 * with a stability guarantee, and it applies in place - so changing a
 * transform updates the running viewer instead of reloading it.
 */

import {
  makeLayer,
  setupDefaultViewer,
  StatusMessage,
} from "./neuroglancer/neuroglancer.js";

// This module is loaded as `viewer.js?v=<build>`; passing the same tag on
// means a rebuilt camera.js cannot be served from cache behind a fresh
// viewer.js - two halves of different builds fail in ways that look like
// neither.
const buildTag = new URL(import.meta.url).search;
const { carryCameraOver, centreOnData } = await import(
  `./camera.js${buildTag}`
);
const {
  composeAffine,
  dragAngle,
  fromPhysicalMatrix,
  pickDragTarget,
  pixelOffset,
  planeRotation,
  rotationMatrix,
  toPhysical,
  toPhysicalMatrix,
  translateMatrix,
  translationForDrag,
} = await import(`./placement.js${buildTag}`);
const { containsPoint, coveragePolygon, layerGeometry, layerSlabs } =
  await import(`./highlight.js${buildTag}`);

//: Events that mean the user is driving the camera. Until one arrives, the
//: viewer keeps the camera on the data as the layers report their bounds.
const TAKEOVER_EVENTS = ["pointerdown", "wheel", "keydown"];

//: Neuroglancer action a placement drag raises while manual placement is on.
//: Bound on the app's own slice-view map, so it applies in the cross-sections
//: only. Modifiers are matched exactly, so the two gestures need a binding
//: each - adding control to the translation binding would not match it.
//:
//: Alt moves, control+alt turns. Alt displaces `move-annotation`, which needs
//: an annotation layer to do anything; control+alt is unbound in a slice view.
//: A control+alt *drag* and a control *click* are different gestures, so the
//: two can share the modifier without ever meaning the same thing.
const MOVE_LAYER_ACTION = "mvs-move-layer";
const MOVE_LAYER_EVENTS = [
  "at:alt+mousedown0",
  "at:control+alt+mousedown0",
];

//: Bound while layers can be picked, to an action nothing listens for. The
//: pick itself is handled from the pointer events - only there can a press
//: that turned into a pan be told from a click - so all these bindings do is
//: take the gesture off Neuroglancer's `annotate`, which answers a ctrl+click
//: with "the annotate command requires a layer to be selected".
const PICK_LAYER_ACTION = "mvs-pick-layer";
const PICK_LAYER_EVENTS = ["at:control+mousedown0", "at:meta+mousedown0"];

//: Viewport pixels a rotation's plane basis is measured over. See `#planeBasis`.
const BASIS_SPAN = 256;

//: How far the pointer may travel between press and release and still count as
//: a click. A plain drag in a cross-section pans the camera, and panning must
//: not end in a selection.
const CLICK_SLOP_PX = 4;

const POSITIONAL_COLOR_SHADER = `#uicontrol invlerp contrast
#uicontrol vec3 color color
void main() {
  float contrast_value = contrast();
  if (VOLUME_RENDERING) {
    emitRGBA(vec4(color * contrast_value, contrast_value));
  } else {
    emitRGB(color * contrast_value);
  }
}
`;

/** Wraps one mounted Neuroglancer instance. */
export class NeuroglancerViewer {
  #viewer = null;
  #target = null;
  #stateListeners = new Set();
  #positionListeners = new Set();
  #onStateChanged = null;
  #onPositionChanged = null;
  #onCoordinateSpaceChanged = null;
  #onToolPalettesChanged = null;
  #applying = false;
  #placing = false;
  #cameraPlaced = false;
  #onTakeover = null;
  #lastNames = null;
  #lastPosition = null;
  #positionalBackups = new Map();
  #placement = null;
  #onMoveLayer = null;
  #drag = null;
  #interaction = null;
  #highlights = new Map();
  #overlay = null;
  #drawnHighlights = null;
  #frame = null;
  #pointerInside = false;
  #hoveredUrl = null;
  #press = null;
  #onPointerEnter = null;
  #onPointerLeave = null;
  #onPointerDown = null;
  #onPointerUp = null;

  /**
   * Mount a viewer into `target`.
   *
   * Safe to call once; call `dispose()` before mounting another.
   */
  mount(target) {
    if (this.#viewer) throw new Error("the viewer is already mounted");

    this.#target = target;
    // Keep Neuroglancer's standard layer and shader controls available. The
    // panels themselves start closed, but users can open either one from the
    // viewer toolbar whenever they need the full controls.
    this.#viewer = setupDefaultViewer({
      target,
      showToolPaletteButton: false,
    });

    // Neuroglancer opens a tool palette on its own, so hiding the button that
    // opens one is not enough - see `#hideSidePanels`.
    this.#onToolPalettesChanged = () => this.#hideSidePanels();
    this.#viewer.toolPalettes.changedShallow.add(this.#onToolPalettesChanged);
    this.#hideSidePanels();

    this.#mountOverlay();

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

    // Cleared first: a teardown is not a placement the app should be told to
    // save. `reset()` re-mounts, and the app re-applies its placement then.
    this.#placement = null;
    this.#unbindManualPlacement();
    this.#unbindLayerInteraction();
    if (this.#frame !== null) cancelAnimationFrame(this.#frame);
    this.#frame = null;
    this.#highlights = new Map();
    this.#overlay = null;
    this.#drawnHighlights = null;
    this.#viewer.state.changed.remove(this.#onStateChanged);
    this.#viewer.navigationState.position.changed.remove(
      this.#onPositionChanged,
    );
    this.#viewer.navigationState.coordinateSpace.changed.remove(
      this.#onCoordinateSpaceChanged,
    );
    this.#viewer.toolPalettes.changedShallow.remove(
      this.#onToolPalettesChanged,
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
    this.#onToolPalettesChanged = null;
    this.#lastNames = null;
    this.#lastPosition = null;
    this.#positionalBackups.clear();
    if (this.#target) this.#target.replaceChildren();
    this.#target = null;
  }

  #require() {
    if (!this.#viewer) throw new Error("the viewer is not mounted");
    return this.#viewer;
  }

  /**
   * Keep Neuroglancer's own side panels closed.
   *
   * The layer and shader panels start closed but stay reachable from the
   * viewer toolbar. The tool palette is different: `showToolPaletteButton`
   * only removes the button that opens one, and Neuroglancer adds a "Shader
   * controls" palette *by itself* every time a multi-channel image finishes
   * loading - docked to the left, over the data. This app has its own channel
   * controls, so that panel is closed again as it appears.
   *
   * Closed rather than deleted: Neuroglancer only adds the palette when no
   * palette with that query exists yet, so the closed one is what stops it
   * from returning with the next multi-channel layer.
   */
  #hideSidePanels() {
    if (!this.#viewer) return;
    this.#viewer.selectedLayer.visible = false;
    this.#viewer.layerListPanelState.location.visible = false;
    for (const palette of this.#viewer.toolPalettes.palettes) {
      palette.location.visible = false;
    }
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
      this.#hideSidePanels();
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
   * as one layer per channel, so every match is returned. Pass `channels` -
   * a set of channel indices - to keep only some of them, which is how a
   * placement restricted to one channel moves only that channel's layer.
   */
  #sourcesReading(url, channels = null) {
    const sources = [];
    for (const managed of this.#viewer.layerManager.managedLayers) {
      if (channels) {
        const channel = this.#channelIndex(managed);
        // A layer with no channel axis is the whole image, so it belongs to
        // every selection rather than to none.
        if (channel !== null && !channels.has(channel)) continue;
      }
      for (const dataSource of managed.layer?.dataSources ?? []) {
        if (dataSource.spec?.url === url) sources.push(dataSource);
      }
    }
    return sources;
  }

  /**
   * Aim each channel of a layer separately.
   *
   * A Neuroglancer source transform is one matrix and a layer is one channel,
   * so a transform that varies over channel arrives beside the viewer state
   * rather than inside it - see `Session.channel_transforms`. `transforms`
   * maps a URL to `{channelIndex: spec}`; channels left out keep whatever the
   * layer specification gave them.
   *
   * Silent about a URL nothing reads: this is applied after a state has been
   * handed over, and a layer may still be loading.
   */
  setChannelTransforms(transforms) {
    this.#require();

    for (const [url, byChannel] of Object.entries(transforms || {})) {
      for (const [channel, transform] of Object.entries(byChannel || {})) {
        const sources = this.#sourcesReading(url, new Set([Number(channel)]));
        for (const dataSource of sources) applyTransform(dataSource, transform);
      }
    }
  }

  /**
   * Add layers to the running viewer, leaving everything else as it is.
   *
   * Going through the viewer state instead would mean restoring a `layers`
   * array, and that clears the layer list and builds every layer again: the
   * chosen layout, the selected layer and each existing layer's shader and
   * contrast range are all replaced. Adding the fused preview is not a reason
   * to lose any of that.
   *
   * Each spec is a Neuroglancer layer specification, exactly as it appears in
   * the `layers` array of a viewer state.
   */
  addLayers(specs) {
    const viewer = this.#require();

    for (const spec of specs) {
      const layer = makeLayer(viewer.layerSpecification, spec.name, spec);
      viewer.layerSpecification.add(layer);
    }
  }

  /**
   * Remove every layer reading from one of `urls`.
   *
   * Silent about a URL that no layer reads: the caller is describing what the
   * viewer should end up showing, and a layer already gone is not a problem.
   */
  removeLayers(urls) {
    const viewer = this.#require();
    const wanted = new Set(urls);

    // Collected first: removing mutates the list being walked.
    const doomed = viewer.layerManager.managedLayers.filter((managed) =>
      (managed.layer?.dataSources ?? []).some((dataSource) =>
        wanted.has(dataSource.spec?.url),
      ),
    );

    for (const managed of doomed) {
      viewer.layerManager.removeManagedLayer(managed);
    }
    return doomed.length;
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

  // -------------------------------------------------------------------
  // Manual placement
  // -------------------------------------------------------------------

  /**
   * Let the user place tiles by hand.
   *
   * Pass null to switch it off. While it is on, in any *orthogonal* panel:
   *
   *   - ctrl+drag moves one layer within the plane of that panel;
   *   - ctrl+alt+drag turns it about its own centre, in the same plane.
   *
   * The perspective panel is left alone, since a drag across a projection does
   * not name a position in the volume. The layer follows the pointer as it
   * goes - the transform of its loaded source is what changes, so nothing is
   * rebuilt and nothing is refetched - and `onDragEnd` fires once, when the
   * pointer is released, which is when the app has a placement worth saving.
   *
   * `placement` carries:
   *   - `movableUrls`: the layers a drag may move, by source URL. Derived
   *     images are simply left out rather than guarded against here.
   *   - `selectedUrls`: the layers the app has selected. Several of them make
   *     a drag act on all of them at once; one breaks a tie where tiles
   *     overlap. Re-supply by calling this again when the selection moves.
   *   - `channels`: channel indices a drag applies to, or null for all. Only
   *     those channels' layers follow the pointer, which is what the app then
   *     asks the session to save. A restriction over *time* has no equivalent
   *     here - a source transform is one matrix for the whole time axis - so
   *     the drag shows the timepoint on screen and the session stores the
   *     range.
   *   - `onDragStart(urls, mode)`, `onDragEnd(urls, mode)`: the boundaries of
   *     one drag, where `mode` is "translate" or "rotate".
   *   - `onRefused(reason)`: a drag that could not be resolved to one layer,
   *     so the app can say why instead of appearing to ignore the user.
   */
  setManualPlacement(placement) {
    const viewer = this.#require();

    if (!placement) {
      // Unbound first, so a drag still in progress ends through the callbacks
      // the app is expecting: the tile has already moved on screen, and that
      // placement is worth saving whatever turned the mode off.
      this.#unbindManualPlacement();
      this.#placement = null;
      return;
    }

    const wasOff = this.#placement === null;
    this.#placement = {
      movableUrls: new Set(placement.movableUrls ?? []),
      selectedUrls: Array.from(placement.selectedUrls ?? []),
      channels: placement.channels ? new Set(placement.channels) : null,
      onDragStart: placement.onDragStart ?? (() => {}),
      onDragEnd: placement.onDragEnd ?? (() => {}),
      onRefused: placement.onRefused ?? (() => {}),
    };

    if (wasOff) {
      // Bound on the app's own slice-view map, whose parent is Neuroglancer's
      // default map at the lowest priority - so this wins over the `annotate`
      // it ships with, and removing it again restores that default. The
      // perspective panel has a separate map and never sees this binding.
      for (const binding of MOVE_LAYER_EVENTS) {
        viewer.inputEventBindings.sliceView.set(binding, {
          action: MOVE_LAYER_ACTION,
          stopPropagation: true,
        });
      }
      // Neuroglancer raises actions as ordinary bubbling DOM events, so no
      // further API is needed to hear about one.
      this.#onMoveLayer = (event) => this.#beginDrag(event);
      this.#target.addEventListener(
        `action:${MOVE_LAYER_ACTION}`,
        this.#onMoveLayer,
      );
    }
  }

  #unbindManualPlacement() {
    this.#endDrag();
    if (!this.#onMoveLayer) return;

    for (const binding of MOVE_LAYER_EVENTS) {
      this.#viewer?.inputEventBindings.sliceView.delete(binding);
    }
    this.#target?.removeEventListener(
      `action:${MOVE_LAYER_ACTION}`,
      this.#onMoveLayer,
    );
    this.#onMoveLayer = null;
  }

  /** True while a tile is being dragged. */
  get dragging() {
    return this.#drag !== null;
  }

  // -------------------------------------------------------------------
  // Hovering and picking layers
  // -------------------------------------------------------------------

  /**
   * Let the user point at and pick layers in the viewer.
   *
   * Separate from `setManualPlacement`, and on whether or not that is: moving
   * a tile is a mode the user turns on, while knowing which tile is under the
   * pointer - and saying "that one" - is how the viewer answers a question the
   * views list otherwise has to be read for.
   *
   * A layer is only ever named when it is the *only* one under the pointer.
   * Where tiles overlap the pointer does not identify one, and a highlight or
   * a selection that guessed would be worse than none: the views list is
   * there, and it is unambiguous.
   *
   * `interaction` carries:
   *   - `urls`: the layers that take part, by source URL.
   *   - `onHover(url | null)`: the layer under the pointer whenever that
   *     changes, so the app can show it wherever else the layer appears.
   *   - `onPick(url, { extend })`: a click on one. `extend` is set for
   *     ctrl / cmd, which the app is expected to treat as adding to or
   *     removing from a selection rather than replacing it.
   *
   * Pass null to switch it off.
   */
  setLayerInteraction(interaction) {
    this.#require();

    if (!interaction) {
      this.#unbindLayerInteraction();
      return;
    }

    const wasOff = this.#interaction === null;
    this.#interaction = {
      urls: new Set(interaction.urls ?? []),
      onHover: interaction.onHover ?? (() => {}),
      onPick: interaction.onPick ?? (() => {}),
    };

    if (wasOff) this.#bindLayerInteraction();
    this.#requestFrame();
  }

  /**
   * Outline layers on top of what is drawn, at one of two strengths.
   *
   * `subtle` is the weight the views list gives a selected row and `verySubtle`
   * the weight it gives one under the pointer, so that the two halves of the
   * app say the same thing about a layer at the same volume. A layer named in
   * both takes the stronger of the two.
   *
   * Addressed by source URL, like every other layer operation here, and drawn
   * from each layer's live bounds - so an outline follows a tile through a
   * placement drag rather than being left behind by it.
   */
  setLayerHighlights(highlights) {
    const wanted = new Map();
    for (const url of highlights?.verySubtle ?? []) wanted.set(url, "very-subtle");
    for (const url of highlights?.subtle ?? []) wanted.set(url, "subtle");

    this.#highlights = wanted;
    this.#requestFrame();
  }

  #bindLayerInteraction() {
    const target = this.#target;

    for (const binding of PICK_LAYER_EVENTS) {
      this.#viewer.inputEventBindings.sliceView.set(binding, {
        action: PICK_LAYER_ACTION,
        stopPropagation: true,
      });
    }

    this.#onPointerEnter = () => {
      this.#pointerInside = true;
      this.#requestFrame();
    };
    this.#onPointerLeave = () => {
      this.#pointerInside = false;
      this.#press = null;
      this.#reportHover(null);
      this.#requestFrame();
    };

    // Pressed and released rather than clicked, so that a press which turns
    // into a pan is not a pick. Neuroglancer claims `mousedown` in the cross-
    // sections and stops it there; `pointerdown` is a different event and
    // reaches us anyway, and the capture phase keeps it that way.
    this.#onPointerDown = (event) => {
      if (event.button !== 0) return;
      // Alt is a placement drag and shift a camera rotation. Neither is a
      // pick, and both would otherwise end in one wherever they were short.
      // Control alone is not excluded: that is the pick that extends.
      if (event.altKey || event.shiftKey) return;
      if (!this.#orthogonalPanelAt(event.target)) return;
      this.#press = {
        x: event.clientX,
        y: event.clientY,
        extend: event.ctrlKey || event.metaKey,
      };
    };

    this.#onPointerUp = (event) => {
      const press = this.#press;
      this.#press = null;
      if (!press || event.button !== 0 || !this.#interaction) return;
      if (
        Math.hypot(event.clientX - press.x, event.clientY - press.y) >
        CLICK_SLOP_PX
      ) {
        return;
      }

      const under = this.#urlsUnderPointer();
      if (under.length === 1) {
        this.#interaction.onPick(under[0], { extend: press.extend });
      }
    };

    target.addEventListener("pointerenter", this.#onPointerEnter);
    target.addEventListener("pointerleave", this.#onPointerLeave);
    target.addEventListener("pointerdown", this.#onPointerDown, {
      capture: true,
    });
    // On the window: a press that is released outside the viewer is not a
    // click, and this is where that is noticed rather than left pending.
    window.addEventListener("pointerup", this.#onPointerUp, { capture: true });
  }

  #unbindLayerInteraction() {
    if (this.#interaction) {
      for (const binding of PICK_LAYER_EVENTS) {
        this.#viewer?.inputEventBindings.sliceView.delete(binding);
      }
    }

    if (this.#onPointerDown) {
      this.#target?.removeEventListener("pointerenter", this.#onPointerEnter);
      this.#target?.removeEventListener("pointerleave", this.#onPointerLeave);
      this.#target?.removeEventListener("pointerdown", this.#onPointerDown, {
        capture: true,
      });
      window.removeEventListener("pointerup", this.#onPointerUp, {
        capture: true,
      });
    }

    this.#onPointerEnter = null;
    this.#onPointerLeave = null;
    this.#onPointerDown = null;
    this.#onPointerUp = null;
    this.#interaction = null;
    this.#press = null;
    this.#pointerInside = false;
    this.#hoveredUrl = null;
  }

  /** The layers taking part in interaction that the pointer is on. */
  #urlsUnderPointer() {
    const interaction = this.#interaction;
    if (!interaction || !interaction.urls.size) return [];

    const pointer = this.#pointerPosition();
    if (!pointer) return [];

    return this.#urlsAtPosition(
      pointer.position,
      pointer.globalNames,
      pointer.globalScales,
      interaction.urls,
    );
  }

  #reportHover(url) {
    if (this.#hoveredUrl === url) return;
    this.#hoveredUrl = url;
    this.#interaction?.onHover(url);
  }

  // -------------------------------------------------------------------
  // The highlight overlay
  // -------------------------------------------------------------------

  #mountOverlay() {
    const overlay = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg",
    );
    overlay.setAttribute("class", "mvs-layer-highlights");
    overlay.setAttribute("aria-hidden", "true");
    this.#target.appendChild(overlay);
    this.#overlay = overlay;
    this.#drawnHighlights = null;
  }

  /**
   * Redraw on every frame the overlay is wanted for.
   *
   * The outlines are computed from the panels' own projection, which changes
   * with every pan, zoom, rotation, layout change, window resize and placement
   * drag. Following each of those signals separately would be a list to keep
   * complete; following the frame clock cannot fall behind any of them. The
   * loop runs only while there is something to draw or the pointer is over the
   * viewer, and a frame that computes the same outlines as the last one
   * touches no DOM.
   */
  #requestFrame() {
    if (this.#frame !== null || !this.#viewer) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = null;
      if (!this.#viewer) return;

      // A drag holds the tile under the pointer by definition; re-deciding
      // what is hovered while one is running says nothing and only flickers.
      if (this.#pointerInside && !this.#drag) {
        const under = this.#urlsUnderPointer();
        this.#reportHover(under.length === 1 ? under[0] : null);
      }
      this.#drawHighlights();

      if (this.#highlights.size || this.#pointerInside) this.#requestFrame();
    });
  }

  #drawHighlights() {
    if (!this.#overlay) return;

    const shapes = this.#highlightShapes();
    // Compared before anything is written: the loop runs at frame rate and a
    // still viewer must not be rebuilding SVG sixty times a second.
    const drawn = JSON.stringify(shapes);
    if (drawn === this.#drawnHighlights) return;
    this.#drawnHighlights = drawn;

    this.#overlay.replaceChildren(
      ...shapes.map(({ strength, points }) => {
        const polygon = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "polygon",
        );
        polygon.setAttribute("class", `mvs-layer-highlight ${strength}`);
        polygon.setAttribute(
          "points",
          points.map(([x, y]) => `${x},${y}`).join(" "),
        );
        return polygon;
      }),
    );
  }

  /** One outline per highlighted layer per cross-section panel it shows in. */
  #highlightShapes() {
    if (!this.#highlights.size) return [];

    const viewer = this.#viewer;
    const space = viewer.navigationState.coordinateSpace.value;
    const position = viewer.navigationState.position.value;
    if (!space?.valid || position?.length !== space.rank) return [];

    const { displayDimensionIndices } =
      viewer.navigationState.pose.displayDimensions.value;
    const scales = this.#displayScales();
    // The centre of every cross-section panel is the navigation position, so
    // that is the point the panel's plane is described around.
    const centre = displayCoordinates(position, displayDimensionIndices).map(
      (value, display) => value * scales[display],
    );

    const globalScales = Array.from(space.scales);
    const geometry = new Map();
    for (const url of this.#highlights.keys()) {
      const layer = this.#layerGeometry(url, Array.from(space.names));
      if (layer) geometry.set(url, layer);
    }
    if (!geometry.size) return [];

    // Which of the three on-screen axes each viewer dimension is drawn as, so
    // a panel's own basis can be restated in the dimensions a layer is placed
    // in. One the panel does not draw simply does not move with the pointer.
    const displayAxisOf = new Map();
    displayDimensionIndices.forEach((global, display) => {
      if (global >= 0) displayAxisOf.set(global, display);
    });

    const targetRect = this.#target.getBoundingClientRect();
    const shapes = [];

    for (const panel of viewer.display.panels) {
      if (!panel.sliceView) continue;
      const rect = panel.element.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;

      const plane = this.#planeBasis(
        panel,
        displayCoordinates(position, displayDimensionIndices),
      );
      const left = rect.left - targetRect.left;
      const top = rect.top - targetRect.top;
      const along = (vector) => (global) => {
        const display = displayAxisOf.get(global);
        return display === undefined ? 0 : vector[display];
      };

      for (const [url, layer] of geometry) {
        const slabs = layerSlabs(
          layer,
          layer.globals.map((global) =>
            displayAxisOf.has(global)
              ? centre[displayAxisOf.get(global)]
              : position[global] * globalScales[global],
          ),
          layer.globals.map(along(plane.u)),
          layer.globals.map(along(plane.v)),
        );
        if (!slabs) continue;

        const points = coveragePolygon({
          width: rect.width,
          height: rect.height,
          slabs,
        });
        if (!points.length) continue;

        shapes.push({
          strength: this.#highlights.get(url),
          points: points.map(([x, y]) => [
            round(x + left),
            round(y + top),
          ]),
        });
      }
    }

    return shapes;
  }

  /**
   * Where a layer is, in terms the viewer's own coordinate space can state.
   *
   * `globals` lists the viewer dimensions the layer is placed in; every array
   * that goes in or comes out is indexed the same way. Two shapes come back,
   * and which one says how faithfully the layer can be described:
   *
   *   - `{ matrix, translation, lower, upper }`: the layer's own box, and the
   *     affine map from its source coordinates into those dimensions in
   *     physical units. This is the *image*, edge for edge, whatever rotation
   *     the tile carries;
   *   - `{ bounds }`: the axis-aligned box the layer's output space reports,
   *     used when the transform mixes those dimensions with ones the viewer
   *     cannot state a position in. An over-covering answer beats none.
   *
   * The outline and the hit test are both built from this, so what is drawn
   * around a tile and what counts as being on it cannot come apart.
   */
  #layerGeometry(url, globalNames) {
    for (const dataSource of this.#sourcesReading(url)) {
      const loadState = dataSource.loadState;
      if (!loadState || loadState.error) continue;

      const transform = loadState.transform.value;
      const outputNames = Array.from(transform.outputSpace.names);

      // The dimensions the two spaces share. A layer-local one - a channel
      // axis the viewer has no position along - is not among them.
      const globals = [];
      const rows = [];
      for (let global = 0; global < globalNames.length; global += 1) {
        const row = outputNames.indexOf(globalNames[global]);
        if (row === -1) continue;
        globals.push(global);
        rows.push(row);
      }
      if (!rows.length) return null;

      return { globals, ...layerGeometry(placementOf(transform), rows) };
    }

    return null;
  }

  /**
   * Start a drag, having worked out which layer it moves.
   *
   * Everything the drag needs is captured here - the panel, the untranslated
   * matrices, the geometry of the space - so that each pointer move is
   * arithmetic on a fixed starting point rather than an accumulation of
   * increments, which would drift.
   */
  #beginDrag(actionEvent) {
    const placement = this.#placement;
    const viewer = this.#viewer;
    if (!placement || !viewer || this.#drag) return;

    const event = actionEvent.detail;
    const panel = this.#orthogonalPanelAt(event.target);
    if (!panel) return;

    const pointer = this.#pointerPosition();
    if (!pointer) return;
    const { position, globalNames, globalScales } = pointer;

    const under = this.#urlsAtPosition(
      position,
      globalNames,
      globalScales,
      placement.movableUrls,
    );
    const { urls, reason } = pickDragTarget(under, placement.selectedUrls);
    if (!urls.length) {
      placement.onRefused(reason);
      return;
    }

    // Only the channels the placement applies to follow the pointer. The rest
    // stay where they are, which is exactly what the session will be asked to
    // store, so the drag shows the result rather than a promise of it.
    const sources = urls.flatMap((url) =>
      this.#sourcesReading(url, placement.channels)
        .filter((dataSource) => dataSource.loadState && !dataSource.loadState.error)
        .map((dataSource) => ({ url, dataSource })),
    );
    if (!sources.length) {
      placement.onRefused("no-channels");
      return;
    }

    const { displayDimensionIndices } =
      viewer.navigationState.pose.displayDimensions.value;
    const origin = displayCoordinates(position, displayDimensionIndices);
    // Control turns the tile instead of moving it - both gestures carry alt.
    // Read from the event rather than from a second action, so the two share
    // one code path up to here: the target is picked the same way either way.
    const mode = event.ctrlKey ? "rotate" : "translate";
    // The tile the pointer grabbed, which is the one a rotation is measured
    // around. Every tile then turns by that angle about its own centre.
    const anchor = urls.find((url) => under.includes(url)) ?? urls[0];

    this.#drag = {
      urls,
      anchor,
      panel,
      mode,
      originX: event.clientX,
      originY: event.clientY,
      globalNames,
      globalScales,
      displayDimensionIndices: Array.from(displayDimensionIndices),
      origin,
      // Captured, not read as the drag goes: a tile's bounds move with it, and
      // a rotation centre recomputed from them would chase its own tail.
      bases: sources.map(({ url, dataSource }) => {
        const { transform } = dataSource.loadState;
        const { outputSpace } = transform.value;
        const outputNames = Array.from(outputSpace.names);
        return {
          url,
          dataSource,
          matrix: Float64Array.from(transform.value.transform),
          outputNames,
          outputScales: Array.from(outputSpace.scales),
          centre: tileCentre(outputSpace),
          axes: Array.from(displayDimensionIndices, (global) =>
            global < 0 ? -1 : outputNames.indexOf(globalNames[global]),
          ),
        };
      }),
      onMove: (moveEvent) => this.#moveDrag(moveEvent),
      onUp: () => this.#endDrag(),
    };

    if (mode === "rotate") {
      const plane = this.#planeBasis(panel, origin);
      // Measured around the grabbed tile: that one follows the pointer, and
      // the rest turn by the same angle about their own centres.
      const grabbed =
        this.#drag.bases.find((base) => base.url === anchor) ??
        this.#drag.bases[0];
      const centre = this.#tileCentreOffset(grabbed, plane, origin);
      if (!centre) {
        this.#drag = null;
        placement.onRefused("uncentred");
        return;
      }
      Object.assign(this.#drag, { plane, centre });
    }

    // Listened for on the window, so a drag that leaves the panel - or the
    // page - still ends, instead of leaving a tile stuck to the pointer.
    window.addEventListener("pointermove", this.#drag.onMove);
    window.addEventListener("pointerup", this.#drag.onUp);
    window.addEventListener("pointercancel", this.#drag.onUp);

    placement.onDragStart(urls, mode);
  }

  #moveDrag(event) {
    const drag = this.#drag;
    if (!drag) return;

    const dx = event.clientX - drag.originX;
    const dy = event.clientY - drag.originY;

    if (drag.mode === "rotate") this.#rotateBy(drag, dx, dy);
    else this.#translateBy(drag, dx, dy);
  }

  #translateBy(drag, dx, dy) {
    const displayDelta = this.#displayDelta(drag, dx, dy);

    for (const base of drag.bases) {
      const { transform } = base.dataSource.loadState;
      const translation = translationForDrag({
        displayDelta,
        displayDimensionIndices: drag.displayDimensionIndices,
        globalNames: drag.globalNames,
        globalScales: drag.globalScales,
        outputNames: base.outputNames,
        outputScales: base.outputScales,
      });
      // Assigning the matrix of a *loaded* source moves the data where it is,
      // leaving the layer, its shader and its contrast range untouched.
      transform.transform = translateMatrix(
        base.matrix,
        transform.value.rank,
        translation,
      );
    }
  }

  /**
   * Turn the tile by the angle the pointer has swept around its centre.
   *
   * The whole drag is one rotation of the matrix captured at mousedown, not a
   * rotation composed onto the last frame's: composing every pointer move
   * would accumulate rounding, and the tile would creep in scale as it turned.
   */
  #rotateBy(drag, dx, dy) {
    const angle = dragAngle(drag.centre, { x: 0, y: 0 }, { x: dx, y: dy });
    const rotation = planeRotation(drag.plane.u, drag.plane.v, angle);

    for (const base of drag.bases) {
      const { transform } = base.dataSource.loadState;
      const rank = transform.value.rank;
      const turn = rotationMatrix({
        rotation,
        centre: base.centre,
        axes: base.axes,
        outputScales: base.outputScales,
        rank,
      });
      // Composed in physical units and converted back. A Neuroglancer matrix
      // mixes them - physical linear coefficients, a translation in output
      // pixels - and that mixture does not survive a matrix product.
      transform.transform = fromPhysicalMatrix(
        composeAffine(
          turn,
          toPhysicalMatrix(base.matrix, rank, base.outputScales),
          rank,
        ),
        rank,
        base.outputScales,
      );
    }
  }

  #endDrag() {
    const drag = this.#drag;
    if (!drag) return;

    window.removeEventListener("pointermove", drag.onMove);
    window.removeEventListener("pointerup", drag.onUp);
    window.removeEventListener("pointercancel", drag.onUp);
    this.#drag = null;

    this.#placement?.onDragEnd(drag.urls, drag.mode);
  }

  /** How far a drag of `dx, dy` viewport pixels reaches, in display units. */
  #displayDelta(drag, dx, dy) {
    const moved = drag.panel.translateDataPointByViewportPixels(
      new Float32Array(3),
      Float32Array.from(drag.origin),
      dx,
      dy,
    );
    return Array.from(moved, (value, i) => value - drag.origin[i]);
  }

  /**
   * The physical vectors one viewport pixel spans, across and down.
   *
   * The panel's projection is affine, so one pixel's worth of each axis
   * describes all of it. In physical units rather than voxels: a rotation
   * applied to anisotropic voxel counts is a shear.
   *
   * Measured over a long step and divided down. Neuroglancer projects through
   * `vec3`, which is float32, so a single pixel next to a coordinate in the
   * hundreds is close to the rounding: taken directly, the two axes come back
   * neither quite orthogonal nor quite equal in length, and the rotation built
   * from them carries that error as a slight scale.
   */
  #planeBasis(panel, origin) {
    const drag = { panel, origin };
    const scales = this.#displayScales();
    const perPixel = (dx, dy) =>
      toPhysical(
        this.#displayDelta(drag, dx * BASIS_SPAN, dy * BASIS_SPAN),
        scales,
      ).map((value) => value / BASIS_SPAN);

    return { u: perPixel(1, 0), v: perPixel(0, 1), scales };
  }

  /** The global scale of each of the three display axes. */
  #displayScales() {
    const space = this.#viewer.navigationState.coordinateSpace.value;
    const { displayDimensionIndices } =
      this.#viewer.navigationState.pose.displayDimensions.value;
    return Array.from({ length: 3 }, (_, display) => {
      const global = displayDimensionIndices[display];
      return global === undefined || global < 0 ? 1 : space.scales[global];
    });
  }

  /**
   * Where a tile's centre sits relative to the pointer, in viewport pixels.
   *
   * The angle a rotation drag turns through is measured around this point, so
   * it is fixed when the drag starts.
   */
  #tileCentreOffset(base, plane, origin) {
    // The centre, as an offset from the pointer, along each display axis and
    // in the physical units the plane basis is expressed in.
    const offset = Array.from({ length: 3 }, (_, display) => {
      const output = base.axes[display];
      if (output === undefined || output < 0) return 0;
      return (
        base.centre[output] * base.outputScales[output] -
        origin[display] * plane.scales[display]
      );
    });

    const pixels = pixelOffset(offset, plane.u, plane.v);
    return Number.isFinite(pixels.x) && Number.isFinite(pixels.y)
      ? pixels
      : null;
  }

  /**
   * The cross-section panel an event happened in, if it was one.
   *
   * Panels are told apart by whether they own a slice view: the perspective
   * panel does not, and a drag there is refused rather than guessed at.
   */
  #orthogonalPanelAt(target) {
    if (!(target instanceof Node)) return null;
    for (const panel of this.#viewer.display.panels) {
      if (!panel.sliceView) continue;
      if (panel.element === target || panel.element?.contains(target)) {
        return panel;
      }
    }
    return null;
  }

  /**
   * Where the pointer is in the global coordinate space, if it is anywhere.
   *
   * Shared by every gesture that has to know what it is on: a drag, a hover
   * and a click all mean the same thing by "the layer under the pointer".
   */
  #pointerPosition() {
    const viewer = this.#viewer;
    const space = viewer?.navigationState.coordinateSpace.value;
    const mouse = viewer?.mouseState;
    if (!space?.valid || !mouse?.active) return null;

    const position = Array.from(
      mouse.unsnappedPosition?.length === space.rank
        ? mouse.unsnappedPosition
        : mouse.position,
    );
    if (position.length !== space.rank) return null;

    return {
      position,
      globalNames: Array.from(space.names),
      globalScales: Array.from(space.scales),
    };
  }

  /**
   * Which of `wanted` cover a global position, in layer order.
   *
   * Judged from where each layer *is* rather than from what was drawn, so a
   * tile counts as being under the pointer even where it is hidden behind
   * another one, or dark. Through the same geometry the outline is drawn from,
   * so the tile a pointer picks is the tile the user sees outlined - which
   * stops being automatic the moment a tile is turned, since the corners of a
   * rotated tile fall outside the upright box around it and vice versa.
   */
  #urlsAtPosition(position, globalNames, globalScales, wanted) {
    const urls = [];

    for (const managed of this.#viewer.layerManager.managedLayers) {
      for (const dataSource of managed.layer?.dataSources ?? []) {
        const url = dataSource.spec?.url;
        if (!wanted.has(url)) continue;
        if (urls.includes(url)) continue;

        const geometry = this.#layerGeometry(url, globalNames);
        if (!geometry) continue;

        const point = geometry.globals.map(
          (global) => position[global] * globalScales[global],
        );
        if (containsPoint(geometry, point)) urls.push(url);
      }
    }

    return urls;
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

  /**
   * Apply the app's view and per-channel checkboxes to existing layers.
   *
   * Neuroglancer opens one managed layer per channel. Channel changes are
   * therefore visibility operations, not a reason to replace the layers (or
   * their coordinate space, camera, shader and computed contrast range).
   */
  setDisplayVisibility(visibility, channelVisibility = {}) {
    const viewer = this.#require();
    let matched = 0;

    for (const managed of viewer.layerManager.managedLayers) {
      const url = (managed.layer?.dataSources ?? [])
        .map((dataSource) => dataSource.spec?.url)
        .find((candidate) => Object.hasOwn(visibility, candidate));
      if (url === undefined) continue;

      const layerChannel = this.#channelIndex(managed);
      const channelVisible =
        layerChannel === null
          ? channelVisibility.default !== false
          : channelVisibility[layerChannel] !== false;
      managed.setVisible(Boolean(visibility[url]) && channelVisible);
      matched += 1;
    }

    return matched;
  }

  /** Apply adjacency colors to input layers without rebuilding any layer. */
  setPositionalColors(colors = null) {
    const managedLayers = this.#require().layerManager.managedLayers;
    const live = new Set(managedLayers);

    for (const managed of managedLayers) {
      const url = (managed.layer?.dataSources ?? [])
        .map((dataSource) => dataSource.spec?.url)
        .find((candidate) => colors && Object.hasOwn(colors, candidate));
      const backup = this.#positionalBackups.get(managed);

      if (url !== undefined) {
        const layer = managed.layer;
        if (!layer?.fragmentMain) continue;
        if (!backup) {
          const colorControl = layer.shaderControlState?.value?.get?.("color");
          const originalColor = colorControl?.trackable?.value;
          this.#positionalBackups.set(managed, {
            shader: layer.fragmentMain.value,
            color: originalColor
              ? Float32Array.from(originalColor)
              : null,
          });
        }
        const limits = this.#contrastControl(managed)?.trackable?.value;
        layer.fragmentMain.value = POSITIONAL_COLOR_SHADER;
        this.#applyShaderDisplay(managed, colors[url], limits);
      } else if (backup) {
        const limits = this.#contrastControl(managed)?.trackable?.value;
        managed.layer.fragmentMain.value = backup.shader;
        this.#applyShaderDisplay(managed, backup.color, limits);
        this.#positionalBackups.delete(managed);
      }
    }

    for (const managed of this.#positionalBackups.keys()) {
      if (!live.has(managed)) this.#positionalBackups.delete(managed);
    }
  }

  /** Set only the image shader's displayed min/max on matching channels. */
  setContrastLimits(visibility, channelIndex, limits) {
    let changed = 0;
    for (const managed of this.#require().layerManager.managedLayers) {
      const readsInput = (managed.layer?.dataSources ?? []).some(
        (dataSource) => Object.hasOwn(visibility, dataSource.spec?.url),
      );
      if (!readsInput) continue;
      const layerChannel = this.#channelIndex(managed);
      if (channelIndex !== null && layerChannel !== channelIndex) {
        continue;
      }

      const apply = () => {
        const control = this.#contrastControl(managed);
        if (!control) return false;
        const currentRange = control.trackable.value?.range;
        const range =
          typeof currentRange?.[0] === "bigint"
            ? Array.from(limits, (value) => BigInt(Math.round(value)))
            : Array.from(limits);
        control.trackable.value = {
          ...control.trackable.value,
          range,
          autoCompute: false,
        };
        return true;
      };
      if (!apply()) {
        // A newly opened image parses its shader asynchronously. Queue the
        // requested values on that layer instead of silently skipping it.
        managed.layer?.shaderControlState?.controls?.changed?.addOnce(apply);
      }
      changed += 1;
    }
    return changed;
  }

  /** Current range and slider window for the selected channel layer(s). */
  getContrastLimits(visibility, channelIndex) {
    const ranges = [];
    const windows = [];

    for (const managed of this.#require().layerManager.managedLayers) {
      const readsInput = (managed.layer?.dataSources ?? []).some(
        (dataSource) => Object.hasOwn(visibility, dataSource.spec?.url),
      );
      if (!readsInput) continue;
      const layerChannel = this.#channelIndex(managed);
      if (channelIndex !== null && layerChannel !== channelIndex) {
        continue;
      }
      const value = this.#contrastControl(managed)?.trackable?.value;
      const range = Array.from(value?.range ?? [], Number);
      const window = Array.from(value?.window ?? [], Number);
      if (range.length === 2 && range.every(Number.isFinite)) ranges.push(range);
      if (window.length === 2 && window.every(Number.isFinite)) windows.push(window);
    }

    if (!ranges.length) return null;
    const bounds = windows.length ? windows : ranges;
    return {
      min: Math.min(...ranges.map((range) => range[0])),
      max: Math.max(...ranges.map((range) => range[1])),
      lower: Math.min(...bounds.map((range) => range[0])),
      upper: Math.max(...bounds.map((range) => range[1])),
    };
  }

  #channelIndex(managed) {
    const space = managed.localCoordinateSpace?.value;
    const names = Array.from(space?.names ?? []);
    const dimension = names.findIndex(
      (name) => String(name).replace(/'+$/, "") === "c",
    );
    if (dimension < 0) return null;
    const position = Number(managed.localPosition?.value?.[dimension]);
    return Number.isFinite(position) ? Math.floor(position) : null;
  }

  #contrastControl(managed) {
    const controls = managed.layer?.shaderControlState?.value;
    if (!controls?.get) return null;
    return controls.get("contrast") ?? controls.get("normalized") ?? null;
  }

  #applyShaderDisplay(managed, color, limits) {
    const apply = () => {
      const controls = managed.layer?.shaderControlState?.value;
      if (!controls?.get) return false;
      if (typeof color === "string") {
        const match = color.match(/^#?([0-9a-f]{6})$/i);
        const control = controls.get("color");
        if (!match || !control) return false;
        const value = Number.parseInt(match[1], 16);
        control.trackable.value = Float32Array.from([
          ((value >> 16) & 255) / 255,
          ((value >> 8) & 255) / 255,
          (value & 255) / 255,
        ]);
      } else if (color && controls.get("color")) {
        controls.get("color").trackable.value = Float32Array.from(color);
      }

      const contrast = this.#contrastControl(managed);
      if (contrast && limits?.range) {
        const current = contrast.trackable.value;
        const range = Array.from(limits.range, Number);
        const window = Array.from(limits.window ?? limits.range, Number);
        const integral = typeof current?.range?.[0] === "bigint";
        contrast.trackable.value = {
          ...current,
          range: integral
            ? range.map((value) => BigInt(Math.round(value)))
            : range,
          window: integral
            ? window.map((value) => BigInt(Math.round(value)))
            : window,
          autoCompute: false,
        };
      }
      return true;
    };

    if (!apply()) {
      managed.layer?.shaderControlState?.controls?.changed?.addOnce(apply);
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

  /**
   * Move to a timepoint, leaving every other axis where it is.
   *
   * Neuroglancer has no notion of a current frame - time is just another axis
   * of the position - so a timepoint is a coordinate like any other. Returns
   * false when the data has no time axis, which is the app's cue that its
   * slider has nothing to drive.
   */
  setTimepoint(index) {
    const dimensions = this.getPositionDimensions();
    const axis = dimensions.indexOf("t");
    if (axis < 0) return false;

    const position = this.getPosition();
    if (position.length !== dimensions.length) return false;
    // The middle of the voxel: Neuroglancer samples at the position, and the
    // integer edge between two frames is ambiguous.
    position[axis] = Number(index) + 0.5;
    this.setPosition(position);
    return true;
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
 * The middle of a layer's data, in its own output coordinates.
 *
 * The bounds are the transformed ones, so this is where the tile is now - and
 * that, not the origin of its voxel grid, is what a rotation turns about.
 */
function tileCentre(outputSpace) {
  const { lowerBounds, upperBounds } = outputSpace.bounds;
  return Array.from(lowerBounds, (lower, i) => {
    const upper = upperBounds[i];
    return Number.isFinite(lower) && Number.isFinite(upper)
      ? (lower + upper) / 2
      : 0;
  });
}

/**
 * The three on-screen coordinates of a global position.
 *
 * A display axis that is not drawn from any dimension - a 2D dataset has no
 * third - contributes nothing, which is what keeps a drag inside the plane.
 */
/** A tenth of a pixel is finer than an outline can show; see `#drawHighlights`. */
function round(value) {
  return Math.round(value * 10) / 10;
}

/**
 * A layer's transform as plain data, for `highlight.js` to reason about.
 *
 * `RenderLayerTransform.transform` maps the layer's *source* coordinates - its
 * voxel grid - into its output space, as a column-major matrix of `rank + 1`
 * rows and `sourceRank + 1` columns. Both ends are in index units; the output
 * space carries the scales that turn its own indices into metres.
 *
 * A registration or a placement drag is a rotation in that matrix. The output
 * space's `bounds` are the axis-aligned box around the *result*, which is why
 * they cannot describe a turned tile: they are the box the tile is inscribed
 * in, not the tile. The source box is the tile, and the matrix is how it is
 * placed - so that is what an outline has to be drawn from.
 */
function placementOf(transform) {
  return {
    rank: transform.rank,
    sourceRank: transform.sourceRank,
    matrix: transform.transform,
    sourceLower: transform.inputSpace.bounds.lowerBounds,
    sourceUpper: transform.inputSpace.bounds.upperBounds,
    outputLower: transform.outputSpace.bounds.lowerBounds,
    outputUpper: transform.outputSpace.bounds.upperBounds,
    outputScales: transform.outputSpace.scales,
  };
}

function displayCoordinates(position, displayDimensionIndices) {
  return Array.from({ length: 3 }, (_, display) => {
    const global = displayDimensionIndices[display];
    return global === undefined || global < 0 ? 0 : position[global];
  });
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
