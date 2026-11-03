"""Sharp 1080p stills of the app for the demo video (screenshots, not a screen recording: no stutter).

    python demo/capture.py          # needs the server on http://localhost:8000

Writes demo/stills/<scene>.png. The progress screens use a mocked /status, so no analysis has to run.
"""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "stills"
BASE = "http://localhost:8000"
GOAL, PHONE = "00000000005700a1", "00000000000004ef"
QUESTION = "Who was marking blue #38 when he got clear, and how far away was he?"
STEPS = ["track", "calibrate", "geometry", "read", "explain", "graphs"]


def status(running: int, times: list[float]) -> dict:
    return {"clip_id": "0123456789abcdef", "ok": True, "running": True, "failed_step": None,
            "steps": [{"step": i + 1, "name": n, "status": "done" if i < running else "running" if i == running else "pending",
                       "elapsed_s": times[i] if i < running else 0.0, "error": None} for i, n in enumerate(STEPS)]}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
        shot = lambda name: page.screenshot(path=str(OUT / f"{name}.png"))  # noqa: E731

        # upload page, mid-upload
        page.goto(f"{BASE}/#/")
        page.wait_for_selector("#drop")
        page.evaluate("""() => { document.getElementById('upload-progress').hidden = false;
          document.getElementById('upload-bar').style.width = '64%';
          document.querySelector('#drop span').textContent = 'Uploading goal.mp4… 64%'; }""")
        shot("upload")

        # analysis in progress, three moments (mocked status)
        for name, running in (("progress_track", 0), ("progress_read", 3), ("progress_explain", 4)):
            body = json.dumps(status(running, [94, 31, 1, 168, 48, 0]))
            page.unroute("**/status")
            page.route("**/api/clips/0123456789abcdef/status", (lambda b: lambda r: r.fulfill(status=200, content_type="application/json", body=b))(body))
            page.goto(f"{BASE}/?p={name}#/clip/0123456789abcdef")
            page.wait_for_selector("#steps .step")
            page.wait_for_timeout(900)
            shot(name)

        def open_clip(clip: str, t: float):
            page.goto(f"{BASE}/?c={clip}#/clip/{clip}")
            page.wait_for_function(f"typeof R !== 'undefined' && R.data && R.data.clip_id === '{clip}'", timeout=60000)
            page.evaluate("t => { const v = document.getElementById('video'); v.muted = true; v.currentTime = t; }", t)
            page.wait_for_function("(() => { const v = document.getElementById('video'); return !v.seeking && v.readyState >= 2; })()", timeout=60000)
            page.wait_for_timeout(600)

        # result page: the top, mid build-up
        open_clip(GOAL, 12.9)
        shot("result_top")
        # the summary and a claim with a measured reason
        pick = page.evaluate("R.data.explanation.claims.findIndex(c => c.evidence.some(e => String(e.id).startsWith('cue_')) && c.evidence.some(e => String(e.id).startsWith('why_')))")
        pick = max(pick, 0)
        page.locator("#claims article .claim-main").nth(pick).click()
        page.evaluate("document.getElementById('video').pause()")
        page.evaluate("document.getElementById('explain-title').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(500)
        shot("why")
        page.evaluate("i => document.querySelectorAll('#claims article')[i].scrollIntoView({block: 'center'})", pick)
        page.wait_for_timeout(400)
        shot("claim")
        # the body cues behind it
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(400)
        shot("cue")
        # what the audit struck
        page.evaluate("const d = document.getElementById('struck'); d.open = true; d.scrollIntoView({block: 'center'})")
        page.wait_for_timeout(400)
        shot("struck")
        # a question with its evidence
        page.evaluate("document.getElementById('ask-form').scrollIntoView({block: 'center'})")
        page.fill("#ask-input", QUESTION)
        page.click("#ask-send")
        page.wait_for_function("document.querySelector('#ask-answer p')", timeout=120000)
        page.evaluate("document.getElementById('ask-form').scrollIntoView({block: 'center'})")
        page.wait_for_timeout(400)
        shot("ask")
        # marking at the cross, then the heatmap and a measured distance
        open_clip(GOAL, 18.6)
        shot("marking")
        page.click("#t-heat"); page.click("#t-heat")
        page.wait_for_timeout(400)
        shot("heatmap")
        # a clip with no goal, explained from positions
        open_clip(PHONE, 10.4)
        shot("phone_top")
        page.evaluate("document.getElementById('explain-title').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(400)
        shot("phone_why")
        # title, results and end cards
        card = """<html><body style="margin:0;width:1280px;height:720px;background:#0F1412;color:#E8EDEA;font-family:'Segoe UI',sans-serif;
          display:flex;flex-direction:column;justify-content:center;align-items:center;gap:18px">{}</body></html>"""
        td = 'style="padding:9px 24px"'
        cards = {
            "title": """<div style="font-size:64px;font-weight:700;letter-spacing:-.02em">Goal Explainer</div>
              <div style="font-size:27px;color:#B9C5BF">One video of a goal in. <span style="color:#F2B33D">Why it happened</span> out.</div>
              <div style="font-size:18px;color:#9AA8A1;margin-top:14px">Meta Model API only: SAM 3.1 + Muse Spark · no training</div>""",
            "results": f"""<div style="font-size:36px;font-weight:700;margin-bottom:6px">Three goals, three matches</div>
              <table style="font-size:20px;border-collapse:collapse">
              <tr style="color:#9AA8A1"><td {td}></td><td {td}>Goal 1</td><td {td}>Goal 2</td><td {td}>Goal 3</td></tr>
              <tr><td {td} style="color:#9AA8A1">Players on screen</td><td {td}>8 + 10</td><td {td}>8 + 9</td><td {td}>10 + 10</td></tr>
              <tr><td {td} style="color:#9AA8A1">Passes found</td><td {td} style="color:#8FD9AE">5</td><td {td} style="color:#F5B0A8">0</td><td {td} style="color:#F5B0A8">1</td></tr>
              <tr><td {td} style="color:#9AA8A1">Goal time</td><td {td}>20.0 s</td><td {td}>22.5 s</td><td {td}>22.3 s</td></tr>
              <tr><td {td} style="color:#9AA8A1">Verdict</td><td {td} style="color:#8FD9AE">build-up + reason</td>
                <td {td} style="color:#F5B0A8">reason at the finish only</td><td {td} style="color:#F5D9A0">goal right, build-up missed</td></tr></table>
              <div style="font-size:17px;color:#9AA8A1;margin-top:10px">Small players on wide panoramas lose passes. The page says less instead of making it up.</div>""",
            "end": """<div style="font-size:58px;font-weight:700">Goal Explainer</div>
              <div style="font-size:24px;color:#B9C5BF">SAM 3.1 + Muse Spark · Meta Model API</div>
              <div style="font-size:22px;color:#F2B33D">github.com/furkankupcu/goal-explainer</div>
              <div style="font-size:15px;color:#9AA8A1;margin-top:20px">Goal clips: SoccerTrack v2 (CC BY 4.0), matches 117093, 118575, 128058</div>""",
        }
        for name, html in cards.items():
            page.set_content(card.format(html))
            shot(name)
        browser.close()
    print("stills:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
