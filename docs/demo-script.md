# Demo video — shot list (≈ 2:30)

Record the screen at 1080p with the app already loaded on a finished run (open the
SoccerTrack goal: `http://localhost:8000/#/clip/<id>`), so nothing has to wait.
Voice-over lines are suggestions; keep them short.

| Time | Screen | Say |
|---|---|---|
| 0:00 | The goal clip playing full screen, no UI | "Pro clubs explain goals with multi-camera tracking. Everyone else has one video. Goal Explainer turns that video into an analyst's explanation." |
| 0:12 | Upload page, then the progress list ticking through the 6 steps (speed it up) | "It uses only Meta's models: SAM 3.1 to track every player and map the pitch, Muse Spark to read what positions can't show, and to write." |
| 0:30 | Result page. Read the summary out loud | "Here is why this goal happened." |
| 0:45 | Click claim 2 (blue #38 pins white #6). Video and top-down view jump; point at the body-cue crop | "Every claim points to a moment and to its evidence. Here Muse Spark read white #6 facing the ball with his back to his own goal." |
| 1:05 | Click the last claim, then the **Goal** marker on the timeline | "Muse Spark watched the clip and found the shot at 19.3 seconds, even though the ball is lost in the crowded box." |
| 1:20 | Turn on **Marking**, play the last 6 seconds: lines on the video and on the pitch, red "free" labels | "Live, who is marking whom: green tight, yellow loose, red free. Three attackers are unmarked as the cross comes in." |
| 1:40 | Turn on **Team shape** for a few seconds, then open the downloaded `graphs.mp4` | "Team shape, and the same view as a video to share." |
| 1:55 | README "How it works" table, then "Measured results" | "It measures first and only then asks the model. Code drops any claim without evidence. And we measure ourselves: calibration is within a metre on a full-pitch phone video; on a narrow view it falls back to four points." |
| 2:15 | The phone clip from the stands with players boxed (10 + 10 + keepers) | "It works on a phone video filmed from the stands, too." |
| 2:25 | Title card: Goal Explainer · SAM 3.1 + Muse Spark · repo link | "Goal Explainer: for everyone who only has one camera." |

Credits line at the end: *Goal clip: SoccerTrack v2 (CC BY 4.0), match 117093.*
