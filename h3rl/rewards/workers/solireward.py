"""SoliReward (CVPR 2026, InternVL3-1B pixel outcome reward models): text alignment (TA-HPQA) and
physics/deformity (physics-deformity-HPQA). Uses the official inference engine with its default prompts;
scalar reward per video from the reward head. Main musubi venv."""
from h3rl.paths import H3_RM
import json, os, sys, tempfile
from h3rl.rewards.worker_base import Worker
CODE = f"{H3_RM}/SoliReward_code"; CK = f"{H3_RM}/SoliReward/pixel_orm"
MODELS = (("soli_ta", "TA-HPQA-InternVL3-1B", "text_alignment"), ("soli_phys", "physics-deformity-HPQA-InternVL3-1B", "phy_deform"))


class Soli(Worker):
    name = "solireward"; batch = 8

    def load(self):
        sys.path.insert(0, CODE)
        import solireward.models.reward_model as srm   # checkpoints ask for flash_attention_2; use eager unless SOLI_ATTN says otherwise
        _load_cfg = srm.load_config_from_json
        def _cfg(path):
            c = _load_cfg(path); c.setdefault("model_args", {})["attn_implementation"] = os.environ.get("SOLI_ATTN", "eager"); return c
        srm.load_config_from_json = _cfg
        from solireward.inference import InferenceArguments, RewardModelInference
        self.eng = {}
        for key, ck, task in MODELS:
            args = InferenceArguments(model_name_or_path=f"{CK}/{ck}", reward_model_task_type=task, batch_size=self.batch,
                                      device="cuda", dtype="bf16", num_workers=0, use_dataloader=False)
            self.eng[key] = RewardModelInference(args)

    def score(self, reqs):
        data = [{"video_path": r["mp4"], "prompt": r["prompt"], "key": r["key"]} for r in reqs]
        res = {r["key"]: {} for r in reqs}
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=self.tmp, delete=False) as f:
            json.dump(data, f); path = f.name
        for key, eng in self.eng.items():
            for item in eng.infer_from_file(input_file=path, output_file=None, show_progress=False):
                res[item["key"]][key] = float(item["score"])
        os.remove(path)
        return [{"key": r["key"], "scores": res[r["key"]], "error": None if len(res[r["key"]]) == 2 else "missing"} for r in reqs]


if __name__ == "__main__": Soli().run()
