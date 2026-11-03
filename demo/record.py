"""Record the demo: drive the running app in Chrome at 1080p, then add the narration.

    python demo/record.py          # needs the server on http://localhost:8000

Writes demo/playwhy-demo.mp4. Narration WAVs come from demo/tts.ps1.
"""
import json
import subprocess
import time
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
BASE = "http://localhost:8000"
GOAL = "00000000005700a1"
PHONE = "00000000000004ef"  # 1080p copy of the phone run: 4K made the recorder lag
QUESTION = "Who was marking blue #38 when he got clear, and how far away was he?"
LINES = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))


def length(name: str) -> float:
    with wave.open(str(HERE / f"vo_{name}.wav")) as w:
        return w.getnframes() / w.getframerate()


CARD = """<html><body style="margin:0;width:1920px;height:1080px;background:#0F1412;color:#E8EDEA;
font-family:'Segoe UI',sans-serif;zoom:.6667;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:28px">{}</body></html>"""

CAPTION_JS = """t => {
  let c = document.getElementById('demo-caption');
  if (!c) {
    c = document.createElement('div'); c.id = 'demo-caption';
    c.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);max-width:1000px;padding:10px 18px;' +
      'background:rgba(10,14,12,.86);color:#fff;font:500 21px/1.35 "Segoe UI",sans-serif;border-radius:12px;z-index:99999;' +
      'text-align:center;box-shadow:0 6px 24px rgba(0,0,0,.4)';
    document.body.appendChild(c);
  }
  c.style.zoom = 1 / (parseFloat(getComputedStyle(document.body).zoom) || 1);  // cards are zoomed out
  c.textContent = t; c.style.display = t ? 'block' : 'none';
}"""


def main() -> None:
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    for old in raw.glob("*.webm"):
        old.unlink()
    marks: list[tuple[str, float]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", args=["--autoplay-policy=no-user-gesture-required"])
        # a 1280 px page scaled to 1080p: at 1920 px the app is a narrow column with tiny text
        ctx = browser.new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(raw),
                                  record_video_size={"width": 1280, "height": 720})
        t0 = time.monotonic()
        page = ctx.new_page()

        def scene(name: str, caption: str | None = None, extra: float = 0.8):
            """Start a scene: log its time, show its caption. Returns when it should end."""
            marks.append((name, time.monotonic() - t0))
            page.evaluate(CAPTION_JS, caption if caption is not None else LINES[name])
            return time.monotonic() + length(name) + extra

        def hold(until: float):
            page.wait_for_timeout(max(0, (until - time.monotonic()) * 1000))

        def play_from(t: float, play: bool = True):
            page.evaluate("t => { const v = document.getElementById('video'); v.muted = true; v.pause(); v.currentTime = t; }", t)
            page.wait_for_function("(() => { const v = document.getElementById('video'); return !v.seeking && v.readyState >= 3; })()", timeout=60000)
            if play:
                page.evaluate("document.getElementById('video').play()")

        def open_clip(clip: str):
            page.goto(f"{BASE}/?r={clip}#/clip/{clip}")  # a full load: cards replace the app's document
            page.wait_for_function("typeof R !== 'undefined' && R.data && R.data.clip_id === '" + clip + "'", timeout=60000)
            page.wait_for_function("document.getElementById('video').readyState >= 2", timeout=60000)

        # 1 title
        page.set_content(CARD.format("""<div style="font-size:96px;font-weight:700;letter-spacing:-.02em">PlayWhy</div>
          <div style="font-size:40px;color:#B9C5BF">One video of a goal in. <span style="color:#F2B33D">Why it happened</span> out.</div>
          <div style="font-size:26px;color:#9AA8A1;margin-top:20px">Meta Model API only: SAM 3.1 + Muse Spark · no training</div>"""))
        hold(scene("title", extra=1.0))

        # 2 the raw goal
        page.set_content(CARD.format(f"""<video id="v" src="{BASE}/api/clips/{GOAL}/video" muted autoplay
          style="width:1800px;max-height:860px;object-fit:contain"></video>"""))
        page.wait_for_function("document.getElementById('v').readyState >= 2", timeout=60000)
        page.evaluate("document.getElementById('v').currentTime = 9")
        hold(scene("clip"))

        # 3 the result page, video playing from the build-up
        open_clip(GOAL)
        play_from(6.5)
        hold(scene("result"))

        # 4 a claim with a measured reason
        page.evaluate("document.getElementById('video').pause()")
        page.locator("#claims article").nth(1).scroll_into_view_if_needed()
        page.locator("#claims article .claim-main").nth(1).click()
        page.evaluate("""() => { const a = document.querySelectorAll('#claims article')[1];
          a.scrollIntoView({block: 'center'});
          a.querySelectorAll('.ev').forEach(b => { if (b.textContent.startsWith('Measured')) b.style.outline = '3px solid #F2B33D'; }); }""")
        hold(scene("why", extra=1.2))

        # 5 the body cue behind it
        page.evaluate("document.getElementById('cues').scrollIntoView({block: 'center'})")
        page.evaluate("document.querySelectorAll('.cue').forEach(c => c.style.outline = '3px solid #F2B33D')")
        hold(scene("cue"))

        # 6 what the audit struck
        page.evaluate("const d = document.getElementById('struck'); d.open = true; d.scrollIntoView({block: 'center'})")
        hold(scene("struck"))

        # 7 ask a question (answer cached by a warm-up call, so no waiting on camera)
        page.evaluate("document.getElementById('ask-form').scrollIntoView({block: 'center'})")
        end = scene("ask", extra=4.0)
        page.locator("#ask-input").click()
        page.keyboard.type(QUESTION, delay=28)
        page.locator("#ask-send").click()
        page.wait_for_function("document.querySelector('#ask-answer p')", timeout=90000)
        page.evaluate("document.getElementById('ask-form').scrollIntoView({block: 'center'})")
        hold(end)

        # 8 marking live, then where each team played
        page.evaluate("window.scrollTo(0, 0)")
        play_from(14.5)
        end = scene("marking", extra=1.5)
        page.wait_for_timeout(5200)
        page.evaluate("document.getElementById('video').pause()")
        page.locator("#t-heat").click()
        hold(end)
        page.locator("#t-marks").click()

        # 9 commentary: the browser voice is not in the recording, so the narration says the line
        end = scene("commentary", extra=0.2)
        page.locator("#cast").click()
        hold(end)
        end = scene("cast", caption='"' + LINES["cast"] + '"', extra=1.2)
        hold(end)
        page.locator("#cast").click()

        # 10 three goals, honestly
        page.set_content(CARD.format("""<div style="font-size:54px;font-weight:700">Three goals, three matches</div>
          <table style="font-size:30px;border-collapse:collapse;color:#E8EDEA">
          <tr style="color:#9AA8A1"><td style="padding:12px 34px"></td><td style="padding:12px 34px">Goal 1</td><td style="padding:12px 34px">Goal 2</td><td style="padding:12px 34px">Goal 3</td></tr>
          <tr><td style="padding:12px 34px;color:#9AA8A1">Players on screen</td><td style="padding:12px 34px">8 + 10</td><td style="padding:12px 34px">8 + 9</td><td style="padding:12px 34px">10 + 10</td></tr>
          <tr><td style="padding:12px 34px;color:#9AA8A1">Passes found</td><td style="padding:12px 34px">5</td><td style="padding:12px 34px">0</td><td style="padding:12px 34px">1</td></tr>
          <tr><td style="padding:12px 34px;color:#9AA8A1">Goal time</td><td style="padding:12px 34px">20.0 s</td><td style="padding:12px 34px">22.5 s</td><td style="padding:12px 34px">22.3 s</td></tr>
          <tr><td style="padding:12px 34px;color:#9AA8A1">Verdict</td><td style="padding:12px 34px;color:#0ca30c">build-up + reason</td>
            <td style="padding:12px 34px;color:#fab219">reason at the finish only</td><td style="padding:12px 34px;color:#fab219">goal right, build-up missed</td></tr>
          </table>"""))
        hold(scene("results"))

        # 11 a phone video from the stands
        open_clip(PHONE)
        play_from(4)
        end = scene("phone", extra=1.2)
        page.wait_for_timeout(4500)
        page.evaluate("document.getElementById('explain-title').scrollIntoView({block: 'start'}); window.scrollBy(0, -20)")
        hold(end)

        # 12 end card
        page.set_content(CARD.format("""<div style="font-size:88px;font-weight:700">PlayWhy</div>
          <div style="font-size:36px;color:#B9C5BF">SAM 3.1 + Muse Spark · Meta Model API</div>
          <div style="font-size:32px;color:#F2B33D">github.com/furkankupcu/playwhy</div>
          <div style="font-size:22px;color:#9AA8A1;margin-top:30px">Goal clips: SoccerTrack v2 (CC BY 4.0), matches 117093, 118575, 128058</div>"""))
        hold(scene("end", extra=1.8))
        total = time.monotonic() - t0
        video_path = page.video.path()
        ctx.close()
        browser.close()

    # Narration track: the recorder shows a change ~1 s after it happens, so each line starts 0.9 s in.
    rate = 22050
    track = np.zeros(int((total + 1) * rate), np.int16)
    for name, t in marks:
        with wave.open(str(HERE / f"vo_{name}.wav")) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        i = int((t + 0.9) * rate)
        track[i:i + len(a)] = a[:len(track) - i]
    with wave.open(str(HERE / "narration.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(track.tobytes())
    out = HERE / "playwhy-demo.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
                    "-i", str(HERE / "narration.wav"), "-map", "0:v", "-map", "1:a", "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264", "-crf", "20",
                    "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "160k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)
    (HERE / "marks.json").write_text(json.dumps(marks, indent=1))
    print("wrote", out, round(total, 1), "s")


if __name__ == "__main__":
    main()
