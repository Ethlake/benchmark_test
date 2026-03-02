import json

class SmokeAgent:
    def __init__(self):
        self.stage = 0
        self.mixed_name = None
        self.model_id = None

    def act(self, obs):
        payload = json.loads(obs["payload"])
        # 第一步：加载两个数据集（随便先加载一个，vlm_mix 前 env 会 ensure_full）
        if self.stage == 0:
            self.stage += 1
            return {"tool": "load_dataset", "args": {"dataset_id": "vqav2"}}

        # 第二步：mix（每个 dataset 一个 count）
        if self.stage == 1:
            self.stage += 1
            return {"tool": "vlm_mix", "args": {"sample_counts": "20,20", "sample_mode": "random", "seed": 42}}

        # 第三步：finetune（dataset_id 用 vlm_mix 输出的 combined_name）
        if self.stage == 2:
            # 从上一步结果里取 combined_name
            result_str = payload.get("result", "{}")
            try:
                r = json.loads(result_str)
                self.mixed_name = r["combined_name"]
            except Exception:
                pass
            self.stage += 1
            return {"tool": "submit_finetune", "args": {"dataset_id": self.mixed_name}}

        # 第四步：eval（model_id 用 submit_finetune 返回的 job_id）
        if self.stage == 3:
            result_str = payload.get("result", "{}")
            r = json.loads(result_str)
            self.model_id = r["job_id"]
            self.stage += 1
            return {"tool": "submit_eval", "args": {"model_id": self.model_id}}

        # 兜底
        return {"tool": "load_dataset", "args": {"dataset_id": "vqav2"}}