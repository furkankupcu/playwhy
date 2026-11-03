"""Cut the demo video from the stills and the source videos: no audio, captions burned in.

    python demo/capture.py && python demo/compose.py

Each scene is its own 1080p/30 fps segment with a short fade, then the segments are joined
without re-encoding. Stills and source video play at their own frame rate, so nothing stutters.
"""
import subprocess
import textwrap
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
ROOT = HERE.parent
STILLS, SEG = HERE / "stills", HERE / "segments"
FF = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, FADE = 1920, 1080, 30, 0.35
FONT = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 34)

# (name, seconds, source, caption); source is a still name or (video path, start s)
SCENES = [  # (name, seconds, source, caption, crop as (x, y, width) in still pixels; height follows 16:9)
    ("title", 4.0, "title", "", None),
    ("goal", 10.0, (ROOT / "clips/st_goal_crop.mp4", 10.0), "A goal from a university match, filmed by one panoramic camera. Why did it happen?", None),
    ("upload", 4.5, "upload", "Upload the clip. Say whether it has a goal, or let Muse Spark decide.", None),
    ("progress_track", 2.8, "progress_track", "SAM 3.1 tracks every player and the ball, then maps them to pitch metres.", (300, 120, 1320)),
    ("progress_read", 2.8, "progress_read", "Muse Spark watches the clip and reads the key players' body language.", (300, 120, 1320)),
    ("progress_explain", 2.8, "progress_explain", "Muse Spark writes from measured facts. Code and a second pass check it.", (300, 120, 1320)),
    ("result_top", 6.0, "result_top", "The result: the video with live marking, and a top-down pitch in metres.", None),
    ("why", 7.5, "why", "One headline with the main reason, then short claims. Every number in them is measured.", (0, 0, 1480)),
    ("claim", 8.0, "claim", "Each claim cites its evidence. Green is an edge for the attack, red a weakness, such as a slow reaction.", (0, 160, 1460)),
    ("cue", 6.0, "cue", "What positions cannot show: Muse Spark reads where each player faces, and how fast he reacts.", (0, 300, 1100)),
    ("freeze", 7.0, "freeze", "The turning point, frozen on the video and the pitch: the free attacker's run (green) and what the defender needed.", None),
    ("freeze_zoom", 6.0, "freeze", "Could it be stopped? White #6 needed 6 m/s to reach blue #38: possible. A back-line hole wider than usual is drawn in red.", (940, 300, 980)),
    ("danger", 7.0, "danger", "The attack's edge over the clip. The biggest rise is the turning point, marked in yellow.", (0, 240, 1440)),
    ("struck", 6.0, "struck", "The first draft said too much. Every phrase the data did not support was struck, in red.", (0, 180, 1460)),
    ("ask", 7.0, "ask", "Ask your own question. An answer must cite evidence, or it says the data cannot tell.", (0, 180, 1460)),
    ("marking", 10.0, (ROOT / "runs/st_goal/graphs.mp4", 12.0), "Who marks whom, frame by frame: green tight, yellow loose, red free. Also as an MP4 to share.", None),
    ("shape", 4.5, "shape", "Team shape: open passing lanes, and the ones an opponent blocks.", (940, 120, 980)),
    ("offside", 4.5, "offside", "The offside line at every pass.", (940, 120, 980)),
    ("measure", 4.5, "measure", "Click any two players: the distance between them follows the video.", (940, 120, 980)),
    ("heatmap", 4.5, "heatmap", "Where each team played.", (940, 120, 980)),
    ("calibrate", 6.0, "calibrate", "No lines found? Mark 4 points on a numbered pitch map. The same screen measures our accuracy.", None),
    ("phone_top", 4.5, "phone_top", "No goal in this phone video from the stands.", None),
    ("phone_why", 7.0, "phone_why", "It still explains the move from positions and distances.", (0, 0, 1480)),
    ("results", 8.0, "results", "", None),
    ("next", 7.0, "next", "", None),
    ("end", 5.0, "end", "", None),
]


def caption(text: str) -> Image.Image:
    """A transparent 1080p layer with the caption in a dark box at the bottom."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not text:
        return layer
    d = ImageDraw.Draw(layer)
    lines = textwrap.wrap(text, 70)
    sizes = [d.textbbox((0, 0), ln, font=FONT) for ln in lines]
    tw = max(b[2] - b[0] for b in sizes)
    lh = 48
    bw, bh = tw + 64, lh * len(lines) + 34
    x0, y0 = (W - bw) // 2, H - bh - 46
    d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 16, fill=(10, 14, 12, 225))
    for i, (ln, b) in enumerate(zip(lines, sizes)):
        d.text(((W - (b[2] - b[0])) // 2, y0 + 14 + i * lh), ln, font=FONT, fill=(255, 255, 255, 255))
    return layer


def segment(i: int, name: str, secs: float, source, text: str, crop=None) -> Path:
    out = SEG / f"{i:02d}_{name}.mp4"
    cap = SEG / f"{i:02d}_cap.png"
    caption(text).save(cap)
    fade = f"fade=in:st=0:d={FADE},fade=out:st={secs - FADE:.2f}:d={FADE}"
    if isinstance(source, str):
        still = Image.open(STILLS / f"{source}.png").convert("RGB")
        if crop:  # zoom into the part of the page the caption talks about
            x, y, w = crop
            h = round(w * 9 / 16)
            y = min(y, still.height - h)
            still = still.crop((x, y, x + w, y + h))
        still = still.resize((W, H), Image.LANCZOS)
        still.paste(Image.open(cap), (0, 0), Image.open(cap))
        frame = SEG / f"{i:02d}_frame.png"
        still.save(frame)
        args = ["-loop", "1", "-framerate", str(FPS), "-t", f"{secs}", "-i", str(frame),
                "-vf", f"format=yuv420p,{fade}"]
    else:
        video, start = source
        args = ["-ss", f"{start}", "-t", f"{secs}", "-i", str(video), "-i", str(cap), "-filter_complex",
                f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x0F1412,"
                f"fps={FPS},setsar=1[v];[v][1:v]overlay=0:0,format=yuv420p,{fade}"]
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", *args, "-an", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-r", str(FPS), "-t", f"{secs}", str(out)], check=True)
    return out


def main() -> None:
    SEG.mkdir(exist_ok=True)
    parts = [segment(i, *scene) for i, scene in enumerate(SCENES)]
    lst = SEG / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    out = HERE / "playwhy-demo.mp4"
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, f"{sum(s[1] for s in SCENES):.1f} s")


if __name__ == "__main__":
    main()
