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

  const routes = { SEGMENT, parseRequest, parseFilePath, parseZarrPath };

  if (typeof module !== "undefined" && module.exports) module.exports = routes;
  else root.mvsRoutes = routes;
})(typeof self !== "undefined" ? self : globalThis);
