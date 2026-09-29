"""video_index.py

Create an HTML index of videos under the `videos/` directory. It looks for .mp4 files and optional
.metadata JSON files named <video_basename>.meta.json containing at least:
{
  "return": 123.45,
  "iteration": 100
}

Usage:
python video_index.py --videos-dir videos --out videos/index.html

The generated HTML contains links to the files and a small table sorted by return descending.
"""

import os
import json
import argparse
from pathlib import Path
from html import escape


def find_videos(videos_dir: str):
    out = []
    for root, dirs, files in os.walk(videos_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                path = os.path.join(root, f)
                base = os.path.splitext(path)[0]
                meta_path = base + ".meta.json"
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as fh:
                            meta = json.load(fh)
                    except Exception:
                        meta = {}
                out.append((path, meta))
    return out


def render_index(videos, out_path: str):
    # videos: list[(path, meta)]
    # default return is -inf so unknowns go to bottom
    def score(item):
        meta = item[1]
        return meta.get("return", float("-inf"))

    videos_sorted = sorted(videos, key=score, reverse=True)

    html = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        '<head><meta charset="utf-8"><title>Video Index</title></head>',
        "<body>",
        f"<h1>Video index ({len(videos_sorted)} videos)</h1>",
        '<table border="1" cellpadding="6">',
        "<tr><th>rank</th><th>video</th><th>return</th><th>iteration</th><th>path</th></tr>",
    ]

    for idx, (path, meta) in enumerate(videos_sorted, start=1):
        ret = meta.get("return", "N/A")
        it = meta.get("iteration", "N/A")
        rel = os.path.relpath(path, os.path.dirname(out_path))
        html.append(
            "<tr>"
            f"<td>{idx}</td>"
            f'<td><a href="{escape(rel)}">{escape(os.path.basename(path))}</a></td>'
            f"<td>{escape(str(ret))}</td>"
            f"<td>{escape(str(it))}</td>"
            f"<td>{escape(rel)}</td>"
            "</tr>"
        )

    html.extend(["</table>", "</body>", "</html>"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(html))

    print(f"Wrote index to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--videos-dir", default="videos")
    p.add_argument("--out", default="videos/index.html")
    args = p.parse_args()
    videos = find_videos(args.videos_dir)
    render_index(videos, args.out)
