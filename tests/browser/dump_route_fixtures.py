"""
Generate the route fixtures the service-worker tests check against.

Routes are produced by the real :class:`multiview_stitcher.browser.Session`, so
``routes.test.mjs`` asserts against what Python actually emits rather than
against a hand-written copy of the format.

    python tests/browser/dump_route_fixtures.py
"""

import json
from pathlib import Path

from multiview_stitcher.browser.session import PREVIEW_NAME, Session

OUTPUT = Path(__file__).parent / "fixtures.json"


def main():
    session = Session(session_id="a1b2c3d4e5f6")
    session.generation = 7
    route = session._route(PREVIEW_NAME)

    # Keys zarr asks for: group and array metadata, plus a chunk key whose
    # dimension separator is "/" - the case that makes naive splitting fail.
    keys = [
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "0/.zarray",
        "0/0/0/0/0",
        "1/0/0/3/2",
    ]

    session.generation = 3
    stale_route = session._route(PREVIEW_NAME)

    fixtures = {
        "route": route,
        "zarr_requests": [
            {"route": route, "key": key, "path": f"{route}/{key}"}
            for key in keys
        ],
        "stale_request": {
            "route": stale_route,
            "key": ".zattrs",
            "path": f"{stale_route}/.zattrs",
        },
    }

    OUTPUT.write_text(json.dumps(fixtures, indent=2) + "\n")
    print(f"wrote {OUTPUT} ({len(fixtures['zarr_requests'])} requests)")


if __name__ == "__main__":
    main()
