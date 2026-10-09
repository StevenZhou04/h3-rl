"""Reward terms -> the worker that produces them. A reward config lists terms with weights; the launcher starts
exactly the workers those terms need. Add a reward by writing a Worker in h3rl/rewards/workers/ and registering it here."""

# worker -> how to run it and what it needs
#   env: "main" (the h3rl environment) or "reward" (separate env for models pinned to older transformers)
#   kind: "video" (one request per rollout, default) or "group" (one request per prompt group, see GroupWorker)
#   hf: Hugging Face repos -> local dir under H3RL_REWARDS (None = HF cache);  git: code repos cloned under H3RL_REWARDS
WORKERS = {
    "videoalign": dict(env="reward", hf={"KwaiVGI/VideoReward": "VideoReward", "Qwen/Qwen2-VL-2B-Instruct": None},
                       git={"VideoAlign": ("https://github.com/KwaiVGI/VideoAlign", "219ab9db64c045e5181a2202d11f686439351292")}),
    "hpspp": dict(env="main", hf={"Junjun2333/HPSv3-PlusPlus": "hpsv3pp", "Qwen/Qwen3-VL-8B-Instruct": None},
                  git={"HPSv3-PlusPlus": ("https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus", "9fb1034ecbbe12978ea6feac5aab82d5d00748f2")}),
    "solireward": dict(env="main", hf={"Yukino271828/SoliReward": "SoliReward"},
                       git={"SoliReward_code": ("https://github.com/lian700/SoliReward", "123945876db319b69961a38b3ab24d7859730736")}),
    "unifiedreward": dict(env="main", hf={"CodeGoat24/UnifiedReward-2.0-qwen3vl-8b": "UnifiedReward-2.0-qwen3vl-8b"}, git={}),
    "pairwise_think": dict(env="main", kind="group", hf={"CodeGoat24/UnifiedReward-Think-qwen3vl-8b": "UnifiedReward-Think-qwen3vl-8b"}, git={}),
    "flowmotion": dict(env="main", hf={}, git={}),        # optical flow, CPU only
    "cutcheck": dict(env="main", hf={}, git={}),          # hard-cut detector, CPU only
}

# reward term -> worker.  Higher is better for every term except where the weight is negative in the config.
TERMS = {
    "va_vq": "videoalign", "va_mq": "videoalign", "va_ta": "videoalign",       # VideoAlign visual / motion quality, text alignment
    "hps": "hpspp", "hps_min": "hpspp",                                         # HPSv3++ frame aesthetics (mean, worst frame)
    "soli_ta": "solireward", "soli_phys": "solireward",                         # SoliReward text alignment, physics/deformity
    "ur_align": "unifiedreward", "ur_physics": "unifiedreward", "ur_style": "unifiedreward",
    "flow_motion": "flowmotion", "flow_raw": "flowmotion", "flow_coherence": "flowmotion",   # coherent optical-flow motion
    "cut_free": "cutcheck",
    "think_winrate": "pairwise_think",                                          # group reward: pairwise win rate (VLM judge, both orders)                                                     # penalty for hard cuts in one-shot prompts
}
CPU_WORKERS = {"flowmotion", "cutcheck"}


def workers_for(terms) -> list[str]:
    unknown = [t for t in terms if t not in TERMS]
    if unknown: raise ValueError(f"unknown reward terms {unknown}; known: {sorted(TERMS)}")
    return sorted({TERMS[t] for t in terms})
