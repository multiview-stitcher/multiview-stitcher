/**
 * URL parsing shared by the service worker and its tests.
 *
 * The service worker and Python must agree exactly on how a URL splits into a
 * virtual OME-Zarr route and a key inside it; disagreeing here would show up
 * only as blank tiles in the viewer. Keeping the parsing in one testable place
 * makes that contract checkable from Node.
 *
 * Route format, produced by `multiview_stitcher.browser.session.Session`:
 *
 *     <session id>/g<generation>/<name>.ome.zarr
 *
 * and a full request path looks like:
 *
 *     <app base>/__mvs__/zarr/<route>/<key inside the OME-Zarr>
 */

(function (root) {
  const SEGMENT = "/__mvs__/";

  /** Split a pathname into `{ kind, path }`, or null when we do not own it. */
  function parseRequest(pathname) {
    const index = pathname.indexOf(SEGMENT);
    if (index < 0) return null;

    const rest = pathname.slice(index + SEGMENT.length);
    const separator = rest.indexOf("/");

    return separator < 0
      ? { kind: rest, path: "" }
      : { kind: rest.slice(0, separator), path: rest.slice(separator + 1) };
  }

  /** Split `<mount>/<relative path>` for a local filesystem read. */
  function parseFilePath(path) {
    const separator = path.indexOf("/");
    if (separator <= 0) return null;
    return {
      mount: path.slice(0, separator),
      path: path.slice(separator + 1),
    };
  }

  /**
   * Split `<route>/<key>` for a virtual OME-Zarr read.
   *
   * The route always ends in `.ome.zarr`, and keys may contain further
   * slashes (chunk keys use "/" as the dimension separator), so the split is
   * anchored on that suffix rather than on a slash count.
   */
  function parseZarrPath(path) {
    const marker = ".ome.zarr/";
    const index = path.indexOf(marker);
    if (index < 0) return null;
    return {
      route: path.slice(0, index + marker.length - 1),
      key: path.slice(index + marker.length),
    };
  }

  /**
   * Is a window client one of our app pages, rather than the viewer iframe?
   *
   * The embedded Neuroglancer viewer is a window client too, but it has no
   * handler for our messages, so a request sent to it is never answered.
   */
  function isAppPage(clientUrl, scope) {
    const path = new URL(clientUrl).pathname;
    const viewer = new URL("neuroglancer/", scope).pathname;
    return !path.startsWith(viewer);
  }

  /**
   * Is this key a zarr metadata document rather than chunk data?
   *
   * Metadata is answered by the worker that owns the session, chunk data by
   * whichever worker is free: a layer whose metadata fails to load has
   * nothing to render at all, so it must not depend on a worker being able to
   * reconstruct the session first.
   */
  const METADATA_KEY =
    /(^|\/)(\.zattrs|\.zgroup|\.zarray|\.zmetadata|zarr\.json)$/;

  function isMetadataKey(key) {
    return METADATA_KEY.test(key || "");
  }

  /**
   * Does the tab holding `sessionId` own `route`?
   *
   * One service worker serves every open tab, and it takes the first one that
   * does not decline. A tab that owns no session owns no route either -
   * without this it would answer for another tab's images out of its own
   * empty session, and no amount of fixing the *right* tab would help.
   */
  function ownsRoute(sessionId, route) {
    if (!sessionId || !route) return false;
    return String(route).startsWith(`${sessionId}/`);
  }

  const routes = {
    SEGMENT,
    isMetadataKey,
    ownsRoute,
    parseRequest,
    parseFilePath,
    parseZarrPath,
    isAppPage,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = routes;
  else root.mvsRoutes = routes;
})(typeof self !== "undefined" ? self : globalThis);
