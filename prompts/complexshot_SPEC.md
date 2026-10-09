# Complex-shot prompt spec

How the prompts in complexshot_*.jsonl were written (and how to write more): long, timecoded, multi-shot prompts of
the kind users send to reference-to-video models. They are long, mostly Chinese, multi-shot, 10-15 s, with character references. Write NEW content; do not copy any example.

## Output: JSONL, one object per line, exactly these keys
{"pid": "<family letter><3 digits>, e.g. F1-007",
 "family": "F1|F2|F3|F4|F5",
 "prompt": "<the full prompt, exactly as a user would send it>",
 "reward_prompt": "<faithful complete English translation of prompt (same timecodes, same [cut] markers)>",
 "duration": 10 or 15,
 "segments": [{"t0": 0, "t1": 2.0, "shot": "<shot scale/angle>", "camera": "<camera move>", "cut_before": false}, ...],
 "n_cuts": <number of explicit hard cuts requested ([cut] markers or 分镜 boundaries)>,
 "refs": [{"tag": "@图1", "desc": "<what the reference image shows>"}],
 "has_dialogue": true/false,
 "genre": "<short English tag>", "subject": "<short English tag, unique within your file>"}

Segments must tile [0, duration] contiguously. A segment with cut_before=true starts a new shot (hard cut); otherwise the
camera move continues from the previous segment (shot-scale changes without [cut] are continuous moves, e.g. a push from wide to close).

## Families (write ONLY your assigned family)
F1 "镜头与动作编排" continuous timeline (most common in the eval set):
  opens "生成一段高质量的视频，节奏紧凑，高帧率，60帧/秒，镜头与动作编排：" (or 动画视频 for animation),
  then 5-8 timecoded beats "0-2s：<shot scale + angle>，<scene/action>，<camera move>。2-5s：...". Characters appear as
  "@图2（<short appearance description>）" and the full parenthesised description is REPEATED at every mention.
  Some (about half) have dialogue: lines in quotes inside beats, and at the end a voicestyle block like
  voicestyle:{"character":"<desc>", "voice style":"male voice, middle-aged voice, low pitch, ..."}{...} followed by a sentence that all spoken dialogue must be in <language>.
  Dramatic genres: war/wuxia/historical, thriller, family drama, fantasy. Occasional Traditional Chinese characters are fine (some real prompts use them).
  Mostly continuous moves; 0-2 explicit [cut]s.
F2 beat-by-beat music/dance/animation with hard cuts:
  opens "生成一段高质量动画视频，节奏紧凑，高帧率，60帧/秒，镜头与动作编排： 「", then 6-10 beats of 0.8-1.6 s each separated by " [cut] ",
  each beat = shot scale + subject action on the music beat + camera move ("摄影机快速前推", "镜头上仰", "俯拍"...), lip-synced English lyric lines in quotes in some beats,
  closes "」" then constraints like "角色形象、场景设定和风格必须和输入图保持一致，视频中不能出现水印和字幕。不要生成背景音乐。视频中要保证流畅、自然且符合物理规律的动态化效果，不要出现静态平移的动作".
  Also non-music stylised animation (ink/paper-cut/graphic transitions, "承接上一镜", match cuts). duration 10 (some prompts end at 7-12 s; use 10).
F3 screenplay / 漫剧 format:
  "<场景名> <角色A>-基础形象 <角色B>-<状态形象> ... 【全局设定】<style, colour, 人物保持独立边界, 禁止字幕水印/肢体畸形/闪烁/背景音乐 ...>
   【片段目标】生成由N个连续分镜组成的视频片段 【场景锚点】本片段场景设定在: <场景> 分镜1 5.0s: <dense action + lighting + camera + 环境声> 分镜2 5.0s: ... 【禁止项】<...>"
  N = 2 or 3 (duration 10 or 15). Each 分镜 boundary is a hard cut (n_cuts = N-1). Characters named with -基础形象 / -受伤 etc. suffixes; 1990s-2020s realistic Chinese drama, rural/urban, accidents, confrontations, reunions.
F4 "Timestamped prompts" list (product / food / macro / cinematic object shots):
  "Timestamped prompts\n0-1.2s：<镜头类型>，<景深>。<action with objects in （括号）>。\n1.2-2.0s：切近景特写镜头，..." 8-16 lines, many quick cuts ("切...镜头") and macro camera moves (推进穿过切口, 环绕, 微俯拍).
  Objects in parentheses (（黑色九宫格巧克力盒）). duration 10 or 15.
F5 English R2V prompts (about 1/3 of this family) and short Chinese free-form prompts (2/3):
  English: "Use the uploaded image as the only identity reference. Preserve the exact facial features, hairstyle, ..." then a long cinematic
  description of a candid/handheld/sports/social-media style shot with camera behaviour (e.g. jumbotron, crowd bump, handheld follow), 600-1500 chars.
  Chinese free-form: 150-400 chars, no timecodes, a magical/surreal or VFX event sequence ("三个纸杯缓缓升起...镜头缓缓推进至桌面平视特写..."), or
  "图一是首帧，图二是人物形象，超逼真写实风格，电影级街头运动长镜头，低角度贴地高速跟拍视角（Low-angle tracking shot）。\n【主体与核心动作】：..." style with 【】 headed sections.

## Rules
- Lengths: F1 1000-1800 chars, F2 500-900, F3 500-900, F4 900-1800, F5 as above.
- Every beat names a concrete shot scale/angle and a concrete camera move; actions must be specific and physical.
- Requested camera moves must be demanding but executable (fast push-ins, low tracking, whip pans, crane, orbit, handheld follow, rack focus, slight shake on impact).
- refs: 1-6 per prompt (as in the eval set). Descriptions are short appearance tags.
- Vary genres, settings, eras, art styles (realistic, anime, 3D cartoon, ink, paper-cut, 漫剧). Never reuse a subject.
- Write helper scripts only with filenames prefixed by your family (e.g. F3_batch1.py) in your own folder; append to your own part file only.
