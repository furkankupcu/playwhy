# Goal Explainer

**Upload a short video of a football goal. Read why it happened: who opened the space, who lost their man, and how — with every claim tied to a moment you can see in the clip.**

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026) on the **Meta Model API only**: SAM 3.1 and Muse Spark. No model is trained or fine-tuned.

---

## Not another sports analyzer

Most "AI sports analyzers" send a clip to one model and print what it says. Goal Explainer does the opposite: **the model is never trusted on its own.**

- It **measures first**. SAM 3.1 tracks every player, both keepers, the referee and the ball. Each frame is mapped from pixels to pitch metres. Passes, the offside line and the marking at every frame come from geometry, not from a model's opinion.
- It **reads only where geometry is blind**. Muse Spark looks at crops of the few players who matter at the key moments: body orientation, gaze, a raised arm, a late reaction. Positions alone cannot show these.
- It **writes from evidence**. Muse Spark writes the explanation only from those measured events and readings. Code then drops any claim that cites no evidence, or whose time is not close to the evidence it cites.

Professional clubs get this kind of analysis from multi-camera tracking systems. Everyone else, from amateur coaches to people freezing frames online, has a single video. This project is for them.

## What it produces

For a goal from SoccerTrack v2 (university match, 24 s clip), Muse Spark wrote this from the measured events and its own player readings. Players are named by kit colour and the shirt numbers it could read:

> *Blue scores through a right-sided progression: early separation by blue player 2 from white player 3, depth runs by blue #38 that pin white #6, central recycling via blue player 4 into a long switch to blue #2, and final wide penetration from blue #2 to blue player 8, whose low cross is tapped in by blue player 5 past the white keeper.*

Each claim links to its evidence: a pass, the goal, or a body cue such as *"white #6: ball-oriented, back to own goal, reacting late"*. In the web app, clicking a claim jumps the video and the top-down view to that moment.

The app also draws **who is marking whom** live over the video and on a top-down pitch (tight < 2 m / loose 2–5 m / free > 5 m), plus each team's shape. It also exports the same view as an MP4.

## How it works

```
clip.mp4 → 1 track → 2 calibrate → 3 geometry → 4 read → 5 explain → 6 graphs → web app
```

| Step | Meta model | What happens |
|---|---|---|
| **1 Track** | Muse Spark + SAM 3.1 | Muse Spark reads the shirt colour of both teams, both keepers and the referee from the first frame. SAM 3.1 tracks each one with a colour prompt (`"player in light blue"`). Role prompts like `"goalkeeper"` found nothing, and `"referee"` boxed a player. When two prompts box the same person, Muse Spark looks at the crop and picks the right kit. |
| **2 Calibrate** | SAM 3.1 (+ Muse Spark seed) | SAM 3.1 segments the field, penalty box, centre circle and goals. Candidate fits come from those masks, from Muse Spark keypoints, and from pairs of detected white lines matched to the template's lines. Each is scored on region overlap, white-line alignment on the pitch, and a scale check (players must come out 0.5–1.3 m wide). Four frames spread over the clip are calibrated from scratch. The clearest one becomes the anchor, and camera motion (ORB) carries it forwards and backwards, refitted every 0.5 s. A 4-point manual fallback exists in the UI. |
| **3 Geometry** | — | Pitch positions, possession, passes (the ball must travel ≥ 3 m), shots (the ball must reach the goal mouth), the offside line at each pass, open lanes, lost markers. A lofted ball's ground track is interpolated between touches, because its projection lands far behind the real spot. |
| **4 Read** | Muse Spark (video + images) | Muse Spark watches the whole clip to find the goal and the shot. The ball is often lost in a crowded box, so positions alone miss it. Then, for ~10 key moments, it reads context crops (upscaled, player boxed): orientation, gaze, raised arm, slip, reaction, shirt number, each with a confidence. |
| **5 Explain** | Muse Spark | Writes 3–6 claims and a summary under a JSON schema, citing evidence ids. A code validator drops unsupported or mistimed claims, then a second Muse Spark pass strikes wording the data does not support. It also writes one spoken line per claim. The page lists every struck phrase, and viewers can ask questions: an answer that cites no real evidence id is refused. |
| **6 Graphs** | — | Marking and team-shape graphs per frame (`graphs.json` for the UI, `graphs.mp4` for sharing). |

Everything is cached on disk per request, so re-running a step costs nothing.

## Measured results

**Calibration error**, pixels mapped to the pitch, automatic vs a reference. The pitch is symmetric, so a half-turn fit counts as correct.

| Clip | Reference | Median error |
|---|---|---|
| Phone video from the stands, 4K, most of the pitch visible, 652 frames | earlier fit, verified by eye | **1.9 m** (0.9–2 m for most of the clip; up to 5.7 m in the last seconds, where the camera pans fast) |
| SoccerTrack v2 goal, distorted panorama cropped to the box (one corner of the pitch visible), 600 frames | dataset's hand-placed keypoints | **11.9 m**, steady across the clip (11.3–12.8 m) |

The second row is the weak spot. Before the anchor-frame and line-pair search, single-frame fits on this crop scattered between 15 and 40 m. No single homography fits a stitched panorama exactly: the hand-keypoint reference itself misses its own points by up to 26 px at the edges, and on the anchor frame our fit matches the white lines more closely than the reference does. So part of the 12 m is the reference, not the fit. The SoccerTrack demo still uses the dataset's keypoints through the 4-point manual path, so its positions come from the reference, not from our fit. Reproduce with `python -m eval.calib_eval`.

**Goal clip (SoccerTrack v2, 24 s):**

| | Value |
|---|---|
| Goal found by Muse Spark from the video | shot 19.3 s, goal 20.0 s (dataset label ≈ 21 s, clip start ±1 s) |
| Player readings Muse Spark could make | 21 / 21 (was 0 / 198 before context crops + upscaling) |
| Key moments read | 10 (was 99 before capping) |
| Pipeline time | ~10 min with hand keypoints (track 1.5, read 5, explain 0.3, graph video 4); automatic calibration adds ~4 min (13 min on a 4K phone clip) |
| API cost per clip | ≈ $1 |

**Phone clip (corner kick, 22 s, 4K):** both teams 10 + 10 outfield players, both keepers and the referee found on frame 300. The two passes it detects are the real corner delivery and the ball played back wide.

## Limits

- Spoken commentary uses the browser's built-in voice: the Meta Model API lists no text-to-speech model. Muse Spark writes the words.
- Distance covered follows tracks, so a player whose track breaks counts as two, and calibration error inflates it. Treat it as a comparison within the clip.

- **Narrow views.** With only a corner of the pitch in view, automatic calibration is off by metres (see above). Use the 4-point fallback for those clips. Tight broadcast close-ups are not solved.
- **Small players.** Below ~40 px tall, Muse Spark's readings get vague (confidence ~0.4). Film in 4K from high in the stands if you can.
- **Not verified against ground-truth events.** The explanation's tactical claims have not been checked line by line against the footage.
- **Speed.** ~10 minutes per 20 s clip; not live.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
copy .env.example .env                               # then put your key: MODEL_API_KEY=...
uvicorn server:app --port 8000                       # open http://localhost:8000
```

Command line, one clip end to end:

```bash
python -m goal_explainer.pipeline path/to/clip.mp4 --clip-id my_goal
python -m goal_explainer.pipeline clip.mp4 --clip-id my_goal --skip-to 4   # resume from a step
```

Outputs land in `runs/<clip_id>/`: `tracks.json`, `homography.json`, `pitch.json`, `events.json`, `cues.json`, `explanation.json`, `graphs.json`, `graphs.mp4`. Tests: `python -m pytest`.

## Built during the hackathon

All code in this repository was written during the hackathon, starting 2026-10-27. The git history shows every step. Nothing was carried over from an earlier project.

## Data and credits

- **SoccerTrack v2** (Atom Scott et al.), [huggingface.co/datasets/atomscott/soccertrack-v2](https://huggingface.co/datasets/atomscott/soccertrack-v2), **CC BY 4.0**. The goal clip in the demo is a 24 s crop of match 117093, first half; its hand-placed keypoints are used for calibration and for measuring error. Please cite the SoccerTrack v2 paper (arXiv:2508.01802).
- The phone clip was filmed by the author.
- Pitch, tracking and reasoning models: **Meta Model API** (SAM 3.1, Muse Spark), parsed with Meta's `meta-sam-parser`.

## Repository

```
goal_explainer/   api.py (Meta Model API client + cache) · track.py · calibrate.py · geometry.py
                  read.py · explain.py · graphs.py · pitch.py · pipeline.py
server.py         FastAPI app;  static/index.html  the web UI
eval/             calib_eval.py — calibration error vs hand keypoints
tests/            pytest suite
docs/             application.md (the hackathon application), PLAN.md
```
