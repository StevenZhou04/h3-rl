# 5 s text-to-video+audio prompts (`source: h3rl_t2va`)

Rows of `example_pool.jsonl` (5 s, `t2va`) and `t2va_pool_v2.jsonl`. One JSON object per line:

```json
{"pid": "t2vb0001", "prompt": "...", "task": "t2va", "image": null, "has_audio": true, "audio_prompt": null,
 "camera_move": "<one of the camera moves below, verbatim>", "category": "<one of the categories below>",
 "min_frames": 124, "max_frames": 124, "source": "h3rl_t2va_v2"}
```

## What a prompt must contain (English, 90-170 words, one paragraph)

1. **Subject**: concrete and specific (material, colour, age, wear, clothing, species, breed), never generic.
2. **Setting, time of day and light source**: where, when, and what lights it (e.g. "a single sodium street lamp").
3. **Action**: what moves and how, sized for 5 seconds (one or two beats, never a whole story). For `story` prompts,
   2-3 short beats in an explicit order ("first ..., then ..., finally ...") that fit in 5 s.
4. **Camera**: the camera move written out with its direction, an approximate distance or angle, and its pacing
   (e.g. "easing in at the start and out at the end"). For the locked-off move, say the camera does not move at all.
5. **Continuity constraints**: one continuous take with no cuts; no objects or people added or removed; lens and
   lighting constant; the subject's identity, clothing and position preserved (adapt to what the scene contains).
6. **Sound**: one sentence describing the audio that the scene would produce (ambience, foley, voices without
   intelligible words, music only if a source is visible).

No brand names, no real or famous people, no on-screen text or logos, no violence or gore, nothing sexual,
no children in any risky situation. Write the prompt as plain description, not as instructions to a model.

## Camera moves (use verbatim in `camera_move`)

- locked off completely still on a tripod, no movement of any kind
- a slow continuous clockwise orbit around the subject, staying level
- a slow continuous counterclockwise orbit around the subject, staying level
- a half orbit (about 180 degrees) around the subject, ending opposite where it started
- a steady push-in toward the subject, ending noticeably closer
- a steady pull-back away from the subject, revealing more of the surroundings
- a crane up, rising from below the subject to look down on it
- a crane down, descending from above the subject to its eye level
- a smooth horizontal pan from left to right
- a smooth horizontal pan from right to left
- a smooth tilt downward from the upper part of the scene to the subject
- a smooth tilt upward from the subject to the upper part of the scene
- a tracking shot moving alongside the subject as it moves
- a handheld follow behind the subject, steady but with natural slight sway

## Categories (target share)

| category | share | examples |
|---|---|---|
| `people_action` | 25% | a cook flipping vegetables, a cyclist climbing a hill, two friends laughing on a bench |
| `people_detail` | 10% | hands kneading dough, a face reacting to news, a musician's fingers on strings |
| `animals` | 15% | wild, farm and domestic animals, birds, insects, sea life, behaving naturally |
| `objects_still` | 10% | still lifes and products; the scene itself is motionless (the camera may move) |
| `objects_dynamic` | 10% | machines, liquids, fabric, fire, smoke, falling or rolling objects |
| `nature_landscape` | 10% | weather, water, forests, mountains, deserts, sky |
| `urban_vehicles` | 10% | streets, markets, traffic, trains, boats, aircraft |
| `story` | 10% | 2-3 ordered beats in 5 s: a door opens, a dog runs in, it shakes off rain |

Spread settings across the world (cultures, climates, interiors and exteriors), times of day and weather. Avoid
repeating a subject or a setting within a batch, and avoid defaulting to wood desks, oak tables and desk lamps.
