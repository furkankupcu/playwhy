"""
Step 6 — Player graphs: who marks whom, and each team's shape.

Per frame, from ``pitch.json``:
- marking: each attacker (keepers excluded) → nearest defender, with distance;
- team shape: Delaunay neighbour links per team, flagged ``blocked`` when an
  opponent stands within 1.5 m of the link (a closed passing lane).

Writes ``graphs.json`` (read by the web UI) and renders ``graphs.mp4``
(camera view + both graphs) for the demo video. Both use the same numbers.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from .geometry import point_to_segment_distance

logger = logging.getLogger(__name__)

TIGHT_M, LOOSE_M = 2.0, 5.0     # marking states: tight < 2 m ≤ loose < 5 m ≤ free
SWITCH_M = 1.0                  # a new nearest defender must be this much closer (no flicker)
LANE_BLOCK_M = 1.5              # an opponent this close to a link closes the lane
MAX_LINK_M = 25.0               # longer Delaunay edges are hull artefacts, not passing options


def marking_state(d: float) -> str:
    return "tight" if d < TIGHT_M else "loose" if d < LOOSE_M else "free"


def attacking_team(frames: list[dict[str, Any]]) -> str | None:
    """Team whose player is most often nearest the ball over the clip."""
    votes = {"A": 0, "B": 0}
    for f in frames:
        ball = next((p for p in f["players"] if p["team"] == "ball"), None)
        people = [p for p in f["players"] if p["team"] in votes]
        if ball and people:
            near = min(people, key=lambda p: (p["x_m"] - ball["x_m"]) ** 2 + (p["y_m"] - ball["y_m"]) ** 2)
            votes[near["team"]] += 1
    return max(votes, key=votes.get) if any(votes.values()) else None


def frame_graph(
    people: list[dict[str, Any]],
    keepers: set[int],
    att: str,
    prev_marker: dict[int, int],
) -> dict[str, Any]:
    """Marking links and team-shape links for one frame; updates *prev_marker* in place."""
    dfn = "A" if att == "B" else "B"
    pos = {p["player_id"]: np.array([p["x_m"], p["y_m"]]) for p in people}
    attackers = [p["player_id"] for p in people if p["team"] == att and p["player_id"] not in keepers]
    defenders = [p["player_id"] for p in people if p["team"] == dfn]

    marks = []
    for a in attackers:
        if not defenders:
            break
        dists = {d: float(np.linalg.norm(pos[d] - pos[a])) for d in defenders}
        best = min(dists, key=dists.get)
        cur = prev_marker.get(a)
        if cur in dists and dists[cur] - dists[best] < SWITCH_M:
            best = cur
        prev_marker[a] = best
        marks.append({"att": a, "def": best, "d": round(dists[best], 2), "state": marking_state(dists[best])})

    links = []
    for team, opp in ((att, dfn), (dfn, att)):
        ids = [p["player_id"] for p in people if p["team"] == team]
        if len(ids) < 3:
            continue
        pts = np.array([pos[i] for i in ids])
        opp_pts = [pos[p["player_id"]] for p in people if p["team"] == opp]
        try:
            simplices = Delaunay(pts).simplices
        except Exception:  # noqa: BLE001 — collinear players: no triangulation
            continue
        for j, k in {tuple(sorted((t[n], t[(n + 1) % 3]))) for t in simplices for n in range(3)}:
            if np.linalg.norm(pts[j] - pts[k]) > MAX_LINK_M:
                continue
            blocked = any(point_to_segment_distance(o, pts[j], pts[k]) < LANE_BLOCK_M for o in opp_pts)
            links.append({"team": team, "a": ids[j], "b": ids[k], "blocked": blocked})
    return {"marks": marks, "links": links}


def compute(clip_id: str) -> dict[str, Any]:
    """Graphs for every frame of a clip, from its pitch.json and tracks.json."""
    run_dir = Path("runs") / clip_id
    frames = json.loads((run_dir / "pitch.json").read_text(encoding="utf-8"))
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    keepers = {o["object_id"] for f in tracks["frames"] for o in f["objects"] if o["cls"] == "goalkeeper"}
    att = attacking_team(frames) or "A"
    prev: dict[int, int] = {}
    out = {
        "attacking_team": att,
        "keepers": sorted(keepers),
        "frames": {
            f["frame_idx"]: frame_graph([p for p in f["players"] if p["team"] in ("A", "B")], keepers, att, prev)
            for f in sorted(frames, key=lambda f: f["frame_idx"])
        },
    }
    return out


def run(video_path: Path, clip_id: str, render: bool = True) -> Path:
    """Write ``graphs.json`` and (optionally) render ``graphs.mp4``."""
    run_dir = Path("runs") / clip_id
    graphs = compute(clip_id)
    path = run_dir / "graphs.json"
    path.write_text(json.dumps(graphs), encoding="utf-8")
    logger.info("Graphs for %d frames written to %s (attacking: %s)", len(graphs["frames"]), path, graphs["attacking_team"])
    if render:
        render_video(video_path, clip_id, graphs, run_dir / "graphs.mp4")
    return path


# ---------------------------------------------------------------------------
# MP4 render (camera view + marking graph + team shape)
# ---------------------------------------------------------------------------

BG, SURFACE, GRASS, GRASS_2, LINE = "#0F1412", "#171E1B", "#1F3A2B", "#22402F", "#6E9C80"
TEXT, TEXT_2, MUTED = "#E8EDEA", "#B9C5BF", "#9AA8A1"
TEAM_FILL = {"A": "#F0EFEC", "B": "#1A1A19"}
TEAM_RING = {"A": "#0F1412", "B": "#9AA8A1"}
BALL = "#F2B33D"
# colour + line style, so a state never rests on colour alone (status colours of the dataviz palette)
STATE_STYLE = {
    "tight": ("#0ca30c", "-", "tight  < 2 m"),
    "loose": ("#fab219", (0, (4, 2)), "loose  2–5 m"),
    "free": ("#d03b3b", (0, (1, 1.6)), "free  > 5 m"),
}
DEPTH = 40  # metres from the attacked goal line shown in the pitch panels


def render_video(video_path: Path, clip_id: str, graphs: dict[str, Any], out: Path) -> Path:
    import cv2
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from .track import iter_frames

    run_dir = Path("runs") / clip_id
    pitch = {f["frame_idx"]: f["players"] for f in json.loads((run_dir / "pitch.json").read_text(encoding="utf-8"))}
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    tracks = json.loads((run_dir / "tracks.json").read_text(encoding="utf-8"))
    Hs = [np.array(h["H"]) for h in json.loads((run_dir / "homography.json").read_text(encoding="utf-8"))]
    fps, shirts = tracks["fps"], tracks["shirts"]
    att = graphs["attacking_team"]
    dfn = "A" if att == "B" else "B"
    keepers = set(graphs["keepers"])
    gframes = {int(k): v for k, v in graphs["frames"].items()}
    left = events.get("attacking_direction") == "left"

    for f in ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        if Path(f).is_file():
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = "Segoe UI"
    plt.rcParams["text.color"] = TEXT

    def T(x, y):
        """Pitch metres → panel coords: attacked goal at the bottom."""
        return (68 - y, x) if left else (y, 105 - x)

    def poly(ax, pts, **kw):
        u, v = zip(*(T(x, y) for x, y in pts))
        return ax.plot(u, v, **kw)

    def draw_pitch(ax):
        ax.set_facecolor(BG)
        for k in range(0, 68, 6):
            ax.add_patch(Rectangle((k, -3), 6, DEPTH + 3, color=GRASS if (k // 6) % 2 == 0 else GRASS_2, zorder=0, lw=0))
        kw = dict(color=LINE, lw=1.3, zorder=1)
        g, sgn = (0, 1) if left else (105, -1)
        poly(ax, [(g, 0), (g, 68)], **kw)
        poly(ax, [(g, 0), (g + sgn * DEPTH, 0)], **kw)
        poly(ax, [(g, 68), (g + sgn * DEPTH, 68)], **kw)
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            poly(ax, [(g, 34 - half), (g + sgn * depth, 34 - half), (g + sgn * depth, 34 + half), (g, 34 + half)], **kw)
        t = np.linspace(-0.927, 0.927, 40)
        poly(ax, list(zip(g + sgn * (11 + 9.15 * np.cos(t)), 34 + 9.15 * np.sin(t))), **kw)
        ax.plot(*T(g + sgn * 11, 34), "o", ms=2.5, color=LINE, zorder=1)
        poly(ax, [(g - sgn * 0.9, 30.34), (g - sgn * 0.9, 37.66)], color=TEXT, lw=4, solid_capstyle="butt", zorder=1)
        ax.set_xlim(-1.5, 69.5); ax.set_ylim(-3, DEPTH); ax.set_aspect("equal"); ax.axis("off")

    fig = plt.figure(figsize=(14, 11.4), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.12, wspace=0.04,
                          left=0.03, right=0.97, top=0.89, bottom=0.07)
    ax0 = fig.add_subplot(gs[0, :]); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1, 0]); draw_pitch(ax1)
    ax2 = fig.add_subplot(gs[1, 1]); draw_pitch(ax2)
    for ax, t in ((ax0, "Camera view"), (ax1, "Marking  ·  each attacker → nearest defender"),
                  (ax2, "Team shape  ·  dashed = an opponent blocks the lane")):
        ax.set_title(t, loc="left", fontsize=11, color=TEXT_2, pad=6, fontweight="semibold")
    fig.text(0.03, 0.955, "Who is marking whom", fontsize=20, fontweight="bold", color=TEXT)
    fig.text(0.97, 0.955, "Goal Explainer · SAM 3.1 + Muse Spark", fontsize=9, color=MUTED, ha="right")
    subtitle = fig.text(0.03, 0.925, "", fontsize=11.5, color=TEXT_2)
    handles = [Line2D([], [], color=c, ls=ls, lw=2.4, label=lab) for c, ls, lab in STATE_STYLE.values()] + [
        Line2D([], [], marker="o", ls="", ms=9, mfc=TEAM_FILL[dfn], mec=TEAM_RING[dfn], label=f"{shirts[dfn]} (defending)"),
        Line2D([], [], marker="o", ls="", ms=9, mfc=TEAM_FILL[att], mec=TEAM_RING[att], label=f"{shirts[att]} (attacking)"),
        Line2D([], [], marker="D", ls="", ms=8, mfc=MUTED, mec=BG, label="goalkeeper"),
        Line2D([], [], marker="o", ls="", ms=7, mfc=BALL, mec=BG, label="ball"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=10,
                     bbox_to_anchor=(0.5, 0.012), handlelength=2.6, columnspacing=1.6)
    for t in leg.get_texts():
        t.set_color(TEXT_2)

    dynamic: list = []

    def add(a):
        dynamic.extend(a if isinstance(a, list) else [a])

    def pill(ax, x, y, text, size=8.5):
        add(ax.text(x, y, text, fontsize=size, color=TEXT, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc=SURFACE, ec="none", alpha=0.9)))

    def people_layer(ax, people, ball):
        for p in people:
            u, v = T(p["x_m"], p["y_m"])
            if -3 <= v <= DEPTH:
                keeper = p["player_id"] in keepers
                add(ax.scatter(u, v, s=150 if keeper else 110, marker="D" if keeper else "o",
                               c=TEAM_FILL[p["team"]], edgecolors=TEAM_RING[p["team"]], linewidths=1.6, zorder=5))
        if ball is not None and -3 <= T(ball["x_m"], ball["y_m"])[1] <= DEPTH:
            add(ax.scatter(*T(ball["x_m"], ball["y_m"]), s=55, c=BALL, edgecolors=BG, linewidths=1.5, zorder=6))

    W, Hh = (int(v) for v in fig.canvas.get_width_height())
    writer = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{W}x{Hh}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-movflags", "+faststart", str(out)],
        stdin=subprocess.PIPE,
    )
    cam = None
    try:
        for i, frame in enumerate(iter_frames(video_path)):
            for a in dynamic:
                a.remove()
            dynamic.clear()
            s = 1920 / frame.shape[1]
            img = cv2.cvtColor(cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            if cam is None:
                cam = ax0.imshow(img)
                ax0.set_xlim(0, img.shape[1]); ax0.set_ylim(img.shape[0] * 0.93, img.shape[0] * 0.28)
            else:
                cam.set_data(img)

            rows = pitch.get(i, [])
            people = [p for p in rows if p["team"] in ("A", "B")]
            ball = next((p for p in rows if p["team"] == "ball"), None)
            pos = {p["player_id"]: (p["x_m"], p["y_m"]) for p in people}
            g = gframes.get(i, {"marks": [], "links": []})
            to_px = lambda xy: cv2.perspectiveTransform(np.array([[xy]], np.float64), Hs[i])[0, 0] * s

            for m in g["marks"]:
                a, b = pos[m["att"]], pos[m["def"]]
                col, ls, _ = STATE_STYLE[m["state"]]
                pa, pb = to_px(a), to_px(b)
                add(ax0.plot([pa[0], pb[0]], [pa[1], pb[1]], color=col, ls=ls, lw=2.4, solid_capstyle="round", zorder=3))
                add(poly(ax1, [a, b], color=col, ls=ls, lw=2, solid_capstyle="round", zorder=3))
                if m["state"] == "free":
                    add(ax0.text(pa[0], pa[1] - 22, f"{m['d']:.0f} m", fontsize=9, color=TEXT, ha="center", zorder=4,
                                 bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85)))
                    u, v = T(*a)
                    if v <= DEPTH - 2:
                        pill(ax1, u, v + 2.6, f"{m['d']:.0f} m")
            for link in g["links"]:
                add(poly(ax2, [pos[link["a"]], pos[link["b"]]],
                         color=TEAM_FILL[link["team"]] if link["team"] == dfn else "#7C8A83", lw=1.4,
                         alpha=0.35 if link["blocked"] else 0.85, ls=(0, (2, 2)) if link["blocked"] else "-", zorder=3))
            people_layer(ax1, people, ball)
            people_layer(ax2, people, ball)

            n_free = sum(m["state"] == "free" for m in g["marks"])
            subtitle.set_text(
                f"{i / fps:4.1f} s  ·  {shirts[att]} attack  ·  {n_free} of {len(g['marks'])} attackers "
                f"have no defender within {LOOSE_M:.0f} m" if g["marks"] else f"{i / fps:4.1f} s  ·  no pitch mapping")
            fig.canvas.draw()
            writer.stdin.write(fig.canvas.buffer_rgba().tobytes())
    finally:
        writer.stdin.close()
        writer.wait()
        plt.close(fig)
    logger.info("Graph video written to %s", out)
    return out
