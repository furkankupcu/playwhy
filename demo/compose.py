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
SCENES = [
    ("title", 4.0, "title", ""),
    ("goal", 11.5, (ROOT / "clips/st_goal_crop.mp4", 9.0), "A goal from a university match, filmed by one panoramic camera. Why did it happen?"),
    ("upload", 3.0, "upload", "Upload the clip. Say whether it has a goal, or let Muse Spark decide."),
    ("progress_track", 2.5, "progress_track", "SAM 3.1 tracks every player and the ball, then maps them to pitch metres."),
    ("progress_read", 2.5, "progress_read", "Muse Spark watches the clip and reads the key players' body language."),
    ("progress_explain", 2.5, "progress_explain", "Muse Spark writes from measured facts. Code and a second pass check it."),
    ("result_top", 4.5, "result_top", "The result: the video with live marking, and a top-down pitch in metres."),
    ("why", 6.0, "why", "The reason comes first, in numbers: a gap in the back line wider than usual, extra men around the ball."),
    ("claim", 6.0, "claim", "Every claim cites evidence: green marks an edge for the attack, red a weakness such as a slow reaction."),
    ("cue", 4.5, "cue", "What positions cannot show: Muse Spark reads where each player faces, and how fast he reacts."),
    ("struck", 5.0, "struck", "The first draft said too much. Every phrase the data did not support was struck, in red."),
    ("ask", 5.0, "ask", "Ask your own question. An answer must cite evidence, or it says the data cannot tell."),
    ("marking", 10.0, (ROOT / "runs/st_goal/graphs.mp4", 12.0), "Who marks whom, frame by frame: green tight, yellow loose, red free."),
    ("heatmap", 4.0, "heatmap", "Where each team played, and the distance between any two players."),
    ("phone_top", 3.5, "phone_top", "No goal in this phone video from the stands."),
    ("phone_why", 5.0, "phone_why", "It still explains the move from positions and distances."),
    ("results", 6.0, "results", ""),
    ("end", 4.5, "end", ""),
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


def segment(i: int, name: str, secs: float, source, text: str) -> Path:
    out = SEG / f"{i:02d}_{name}.mp4"
    cap = SEG / f"{i:02d}_cap.png"
    caption(text).save(cap)
    fade = f"fade=in:st=0:d={FADE},fade=out:st={secs - FADE:.2f}:d={FADE}"
    if isinstance(source, str):
        still = Image.open(STILLS / f"{source}.png").convert("RGB").resize((W, H), Image.LANCZOS)
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
    out = HERE / "goal-explainer-demo.mp4"
    subprocess.run([FF, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    print("wrote", out, f"{sum(s[1] for s in SCENES):.1f} s")


if __name__ == "__main__":
    main()
