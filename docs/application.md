# PlayWhy — hackathon application (as submitted, under the working name "Goal Explainer")

I want to build "Goal Explainer": give it a short television clip of a football goal and it explains why the goal happened, the way a good analyst would, with every claim tied to a moment in the clip.

Professional clubs get this analysis from player position data recorded by multi-camera systems in stadiums. Everyone else, from amateur coaches to the analysts who break down goals online by freezing frames and drawing arrows by hand, has only the broadcast. Published tactical models reason over player positions only, and positions alone miss half the story: two defenders can stand in the right place while one is looking the wrong way, has his arm raised for offside, or is turned side-on and cannot react. I want to combine both: where everyone was, and what each of them was actually doing. Recent tactical AI systems for professional clubs predict what will happen next; mine explains what just happened, for everyone else.

The pipeline uses only the Meta Model API. I will not train or fine-tune any model.
1. SAM 3.1 tracks the ball, both teams, the goalkeeper and the pitch markings through the clip from short text prompts.
2. Broadcast cameras pan and zoom, so I estimate the mapping from pixels to pitch metres in every frame from the visible lines. This is the hardest part.
3. From the positions I compute the pass chain, the offside line at each pass, the shot location and angle, and the open passing lanes. This tells the system which moments and which players matter.
4. For those moments, the SAM masks give me a close crop of each key player. Muse Spark looks at the crops and the full clip and reads what positions cannot show: body orientation, where a player is looking, a raised arm, a slip, a late reaction.
5. Muse Spark writes the explanation from both sources: where the move started to work, whose run opened the space, which defender lost his man and how. A simple top-down replay sits beside the text so the reader can follow it.

Scope for ten days: one continuous wide-angle shot, up to twenty seconds, moves that end in or near the penalty area. I will measure the reconstructed positions against a public research dataset of hand-annotated broadcast clips and report the error in metres.

Success means a judge uploads a goal and reads an explanation that points to things they had not noticed, and every one of them is visible in the clip.

---

**Planned vs delivered.** Delivered: everything in steps 1–5, plus measured "why" facts (numbers around the ball, back-line gaps), a second Muse Spark pass that strikes unsupported wording, questions answered from evidence, and clips without a goal. Different from the plan: tested on SoccerTrack v2 panoramas and a phone video from the stands, not on television broadcasts; accuracy was measured against hand-marked points on those (see README, Measured results). The project was renamed PlayWhy because it also explains moves that end without a goal.
