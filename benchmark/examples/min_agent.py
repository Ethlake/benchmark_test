class MinAgent:
    def act(self, obs):
        return {"tool": "think", "args": {"reasoning": "hello from benchmark-only package"}}
