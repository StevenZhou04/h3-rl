"""Blind side-by-side check of two eval arms (e.g. the base model and an RL checkpoint), for a human viewer.

  python -m h3rl.eval.blind_pairs --dir runs/heldout --arms base nft-00060            # writes runs/heldout/blind_base_nft-00060/
  python -m h3rl.eval.blind_pairs --dir runs/heldout --arms base nft-00060 --reveal    # after filling in picks.txt

Pairs the videos of the two arms by key (same prompt, same noise; scripts/heldout_eval.sh writes them to <dir>/<arm>_*/),
shuffles the pairs and puts each pair side by side with a random left/right order: pair_01.mp4, pair_02.mp4, ... The
viewer writes A (left better), B (right better) or = (tie) after every pair number in picks.txt, without opening
.key.json. --reveal maps the picks back to the arms and prints the win counts, a two-sided sign test over the
non-tied pairs, and the pair numbers and prompts of each outcome. Eval videos have no audio, so only the picture is judged."""
import argparse, glob, json, math, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor


def videos(d, arm):
    return {os.path.basename(p)[:-4]: p for p in glob.glob(f"{d}/{arm}_*/*.mp4")}


def side_by_side(left, right, out):
    """Both clips scaled to the left clip's height, a 16 px black gap between them, cut to the shorter clip."""
    h = int(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height",
                                     "-of", "csv=p=0", left]).decode().split()[0])
    fc = f"[0:v]scale=-2:{h},pad=iw+16:ih:0:0:black[l];[1:v]scale=-2:{h}[r];[l][r]hstack=inputs=2:shortest=1[v]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", left, "-i", right, "-filter_complex", fc, "-map", "[v]", "-an",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out], check=True)


def sign_test(wins, losses):
    """Two-sided exact binomial p-value of wins vs losses under p = 1/2 (ties left out)."""
    n = wins + losses
    if n == 0: return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def make(a, out):
    A, B = a.arms; va, vb = videos(a.dir, A), videos(a.dir, B)
    keys = sorted(set(va) & set(vb))
    if not keys: sys.exit(f"no common videos under {a.dir}/{A}_*/ and {a.dir}/{B}_*/")
    if len(keys) < len(set(va) | set(vb)): print(f"note: {len(set(va) ^ set(vb))} videos have no partner and are skipped")
    rng = random.Random(a.seed); rng.shuffle(keys); keys = keys[:a.n] if a.n else keys
    os.makedirs(out, exist_ok=True); prompts = {}
    for m in glob.glob(f"{a.dir}/{A}_*/manifest.jsonl"):
        for l in open(m): r = json.loads(l); prompts[r["key"]] = r.get("prompt", "")
    key, jobs = {}, []
    for i, k in enumerate(keys, 1):
        left, right = (A, B) if rng.random() < 0.5 else (B, A)
        key[f"{i:02d}"] = dict(video=k, left=left, right=right, prompt=prompts.get(k, ""))
        jobs.append(((va if left == A else vb)[k], (vb if right == B else va)[k], f"{out}/pair_{i:02d}.mp4"))
    with ThreadPoolExecutor(a.procs) as ex: list(ex.map(lambda j: side_by_side(*j), jobs))
    json.dump(dict(arms=[A, B], pairs=key), open(f"{out}/.key.json", "w"), indent=1)
    with open(f"{out}/picks.txt", "w") as f:
        f.write("# after each pair number write A (left better), B (right better) or = (tie); a short reason may follow\n")
        f.write("".join(f"{i} \n" for i in key))
    print(f"{len(key)} pairs in {out}/  -> watch pair_*.mp4, fill in {out}/picks.txt, then rerun with --reveal")


def reveal(a, out):
    k = json.load(open(f"{out}/.key.json")); A, B = k["arms"]; pairs = k["pairs"]
    res = {A: [], B: [], "tie": []}; missing = []
    picks = {}
    for l in open(f"{out}/picks.txt"):
        t = l.split(None, 2)
        if t and not t[0].startswith("#"): picks[t[0].zfill(2)] = (t[1].upper() if len(t) > 1 else "", t[2].strip() if len(t) > 2 else "")
    for i, p in pairs.items():
        c, why = picks.get(i, ("", ""))
        if c not in ("A", "B", "="): missing.append(i); continue
        res["tie" if c == "=" else p["left"] if c == "A" else p["right"]].append((i, p["video"], why))
    if missing: print(f"WARNING: no valid pick (A/B/=) for pairs {missing}; they are left out")
    wa, wb, t = len(res[A]), len(res[B]), len(res["tie"]); n = wa + wb + t
    if not n: sys.exit("no picks")
    print(f"{B} wins {wb} / {A} wins {wa} / ties {t}   (n={n})")
    print(f"{B} win rate (ties count 1/2): {(wb + t / 2) / n:.2f}   sign test over non-ties: p={sign_test(wb, wa):.3f}")
    for g in (B, A, "tie"):
        for i, v, why in res[g]: print(f"  [{g if g == 'tie' else g + ' better'}] pair {i} {v}  {why}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval directory holding <arm>_*/ video folders")
    ap.add_argument("--arms", nargs=2, required=True, metavar=("A", "B"), help="reference arm, compared arm")
    ap.add_argument("--out", default=None, help="default <dir>/blind_<A>_<B>")
    ap.add_argument("--n", type=int, default=0, help="number of pairs (default all)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the pair order and the left/right sides")
    ap.add_argument("--procs", type=int, default=8, help="parallel ffmpeg processes")
    ap.add_argument("--reveal", action="store_true", help="score the filled-in picks.txt")
    a = ap.parse_args(); out = a.out or f"{a.dir}/blind_{a.arms[0]}_{a.arms[1]}"
    if a.reveal: reveal(a, out)
    elif os.path.exists(f"{out}/.key.json") and any(l.split()[1:] for l in open(f"{out}/picks.txt") if not l.startswith("#")):
        sys.exit(f"{out}/picks.txt already has picks; rerun with --reveal, or pass another --out")
    else: make(a, out)


if __name__ == "__main__":
    main()
