"""ranks_per_group > 1 (CPU, gloo): a prompt group spread over R ranks must train exactly like one rank holding it.
 (a) every rank of a split group sees all members' rewards; advantages equal the single-rank computation; a team whose
     ranks drew different prompts is refused, on every rank;
 (b) the real NFT.update (advantages, grad_accum / R, gradient averaging, clipping, Adam, EMA) on 2 groups gives the same
     weights with R = 1 (2 ranks, a group each) and R = 2 (4 ranks, each group split over 2)."""
import os, sys, tempfile, random, torch, torch.distributed as dist, torch.multiprocessing as mp
G = 8
REWARDS = {t: [float((7 * t + 3 * k) % 11) for k in range(G)] for t in range(2)}   # group t's rewards by member
XS = {t: [float(t + 0.37 * k) for k in range(G)] for t in range(2)}                 # toy per-sample targets


def worker(rank, world, R, path, out):
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)
    import h3rl.algos.nft as A
    from h3rl.algos.base import TrainContext
    from h3rl.core.nft import group_r
    team, sub = rank // R, rank % R; mine = list(range(sub * G // R, (sub + 1) * G // R))
    w = torch.nn.Parameter(torch.tensor([0.5, -0.25]))
    class Net:
        def set_multiplier(self, m): pass
    class Tr:
        def train(self): pass
    T = TrainContext(models={}, transformer=Tr(), network=Net(), params=[w], schedule=None, device=torch.device("cpu"), rank=rank,
                     world=world, rng=random.Random(100 + rank), infer_steps=8, canvas={"frames": 124, "height": 544, "width": 960})
    acfg = {"lr": 0.05, "group_size": G, "prompts_per_step": 1, "ranks_per_group": R, "grad_accum": 8, "epochs_per_batch": 2,
            "max_grad_norm": 1.0}
    algo = A.NFT(acfg, T)
    def toy_loss(transformer, network, params, old, s, ctx, r_video, r_audio, cfg, schedule, device, rng, loss_scale=1.0):
        x = torch.tensor([s["x"], -s["x"]]); loss = (r_video - 0.5) * ((params[0] - x) ** 2).sum()
        (loss * loss_scale).backward(); return {"loss": float(loss), "fm_video": 0.0}
    A.nft_loss = toy_loss
    samples = [{"gid": f"{team}:0", "member": k, "prompt": {"pid": f"p{team}"}, "ctx": None, "x": XS[team][k],
                "R": {"video": REWARDS[team][k], "audio": None}} for k in mine]
    groups = algo._groups([dict(s) for s in samples])
    full = groups[f"{team}:0"]; ks = sorted(full)
    r_split = dict(zip(ks, group_r([full[k][0] for k in ks], 0.05, 3.0, clip=5.0)))
    r_single = dict(zip(range(G), group_r(REWARDS[team], 0.05, 3.0, clip=5.0)))
    ok_a = ks == list(range(G)) and all(abs(r_split[k] - r_single[k]) < 1e-12 for k in range(G))
    mixed = [dict(s, prompt={"pid": f"p{team}_{sub}"}) for s in samples]       # ranks of a team on different prompts
    try: algo._groups(mixed); ok_mix = R == 1
    except RuntimeError: ok_mix = R > 1
    one_bad = [dict(s, prompt={"pid": f"p{team}_{sub}"}) if team == 0 else s for s in samples]   # only team 0 is inconsistent:
    try: algo._groups(one_bad); ok_all = R == 1                                                  # every rank must stop, not just team 0's
    except RuntimeError: ok_all = R > 1
    algo.update(samples, 0)
    torch.save({"a": ok_a, "mix": ok_mix, "all": ok_all, "w": w.detach().clone(), "old": algo.old[0].clone(), "n": algo.n_updates}, f"{out}/{R}_{rank}.pt")
    dist.destroy_process_group()


def run(R):
    d = tempfile.mkdtemp(); world = 2 * R
    mp.spawn(worker, args=(world, R, f"{d}/pg", d), nprocs=world, join=True)
    return [torch.load(f"{d}/{R}_{r}.pt") for r in range(world)]


if __name__ == "__main__":
    one, two = run(1), run(2); fails = []
    def check(name, ok, info=""):
        print(("  PASS  " if ok else "  FAIL  ") + name + (f"   {info}" if info else "")); fails.extend([] if ok else [name])
    check("(a) split group: every rank gets all members; advantages equal the single-rank ones", all(x["a"] for x in two))
    check("(a) a team whose ranks drew different prompts is refused", all(x["mix"] for x in two) and all(x["mix"] for x in one))
    check("(a) one inconsistent team stops every rank (no rank left waiting in a collective)", all(x["all"] for x in two) and all(x["all"] for x in one))
    check("(b) all ranks hold the same weights after the update", all(torch.equal(x["w"], two[0]["w"]) for x in two) and torch.equal(one[0]["w"], one[1]["w"]))
    check("(b) NFT.update with R = 2 equals R = 1 (weights, EMA copy, optimizer steps)",
          torch.allclose(one[0]["w"], two[0]["w"], atol=1e-6) and torch.allclose(one[0]["old"], two[0]["old"], atol=1e-6) and one[0]["n"] == two[0]["n"],
          f"w R=1 {one[0]['w'].tolist()} R=2 {two[0]['w'].tolist()} steps {one[0]['n']}/{two[0]['n']}")
    check("(b) the update actually moved the weights", not torch.allclose(one[0]["w"], torch.tensor([0.5, -0.25])))
    if fails: sys.exit(1)
    print("ALL PASS")
