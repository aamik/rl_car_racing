"""Video browsing should work before the first recording exists."""

from video_index import find_videos, render_index


def test_empty_index_creates_output_directory(tmp_path):
    output = tmp_path / "videos" / "index.html"
    render_index(find_videos(str(tmp_path / "missing")), str(output))
    assert "Video index (0 videos)" in output.read_text()


def test_index_escapes_names_and_sorts_by_return(tmp_path):
    output = tmp_path / "index.html"
    render_index([
        (str(tmp_path / "low.mp4"), {"return": 1}),
        (str(tmp_path / "<best>.mp4"), {"return": 10}),
    ], str(output))
    html = output.read_text()
    assert "&lt;best&gt;.mp4" in html
    assert html.index("&lt;best&gt;.mp4") < html.index("low.mp4")
