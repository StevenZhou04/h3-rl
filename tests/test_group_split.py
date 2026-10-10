"""ranks_per_group > 1 (CPU, gloo, 4 processes): a prompt group spread over R ranks must train exactly like one rank.
 (a) every rank of a split group sees all members' rewards, and the advantages r equal the single-rank computation;
 (b) one optimizer step with R = 2 (grad_accum / R per rank, gradients averaged over ranks) equals the R = 1 step over
     the same samples."""
import os, sys, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp


def worker(rank, world, path, out):
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=world)
    import h3rl.algos.nft as A
    from h3rl.core.nft import group_r
    from h3rl.core.dist import average_gradients
    G, R = 8, 2; team, sub = rank // R, rank % R; mine = list(range(sub * G // R, (sub + 1) * G // R))
    rewards = {t: [float((7 * t + 3 * k) % 11) for k in range(G)] for t in range(world // R)}   # group t's rewards by member
    class T: pass
    T.world, T.rank = world, rank
    class Cfg: group_size = G
    algo = A.NFT.__new__(A.NFT); algo.T = T; algo.cfg = Cfg(); algo.a = {"ranks_per_group": R}
    samples = [{"gid": f"{team}:0", "member": k, "R": {"video": rewards[team][k], "audio": None}} for k in mine]
    groups = algo._groups(samples)
    full = groups[f"{team}:0"]; ks = sorted(full)
    r_split = dict(zip(ks, group_r([full[k][0] for k in ks], 0.05, 3.0, clip=5.0)))
    r_single = dict(zip(range(G), group_r(rewards[team], 0.05, 3.0, clip=5.0)))
    ok_a = ks == list(range(G)) and all(abs(r_split[k] - r_single[k]) < 1e-12 for k in range(G))
    # (b) toy loss per sample i: 0.5 * (w - x_i)^2 ; R = 1 reference: one rank, samples 0..3, grad_accum 4 -> mean over 4
    xs = torch.tensor([1.0, 2.0, 4.0, 8.0]); w = torch.nn.Parameter(torch.tensor(0.5))
    ref = float(sum((w.detach() - x) for x in xs) / 4)
    # R = 2: ranks 2t and 2t+1 hold samples {0,1} and {2,3} of the same group; grad_accum 4 // 2 = 2 per rank
    w.grad = None
    for x in (xs[:2] if sub == 0 else xs[2:]): (0.5 * (w - x) ** 2 * (1.0 / 2)).backward()
    average_gradients([w], world)       # every team holds the same samples, so the world average equals the team average
    ok_b = abs(float(w.grad) - ref) < 1e-6
    torch.save({"a": ok_a, "b": ok_b, "rs": r_split}, f"{out}/{rank}.pt"); dist.destroy_process_group()


if __name__ == "__main__":
    d = tempfile.mkdtemp(); world = 4
    mp.spawn(worker, args=(world, f"{d}/pg", d), nprocs=world, join=True)
    res = [torch.load(f"{d}/{r}.pt") for r in range(world)]
    fails = []
    for name, key in (("(a) split group: every rank gets all members; advantages equal the single-rank ones", "a"),
                      ("(b) optimizer-step gradient with R = 2 equals R = 1 on the same samples", "b")):
        ok = all(x[key] for x in res); print(("  PASS  " if ok else "  FAIL  ") + name); fails += [] if ok else [name]
    if res[0]["rs"] != res[1]["rs"]: print("  FAIL  team ranks disagree on r"); fails.append("agree")
    else: print("  PASS  both ranks of a team compute identical advantages")
    if fails: sys.exit(1)
    print("ALL PASS")
