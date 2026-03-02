"""
patch_gymnasium.py — Patches ChefsHatGYM source to use gymnasium instead of gym.

Run this ONCE after setup.sh:
    python patch_gymnasium.py
"""

import os

# Files inside the cloned ChefsHatGYM repo to patch
BASE = os.path.join(os.path.dirname(__file__), "ChefsHatGYM", "src", "ChefsHatGym")

files_to_patch = [
    os.path.join(BASE, "env", "__init__.py"),
    os.path.join(BASE, "rewards", "__init__.py"),
    os.path.join(BASE, "env", "ChefsHatEnv.py"),
    os.path.join(BASE, "gameRooms", "chefs_hat_room_local.py"),
    os.path.join(BASE, "gameRooms", "chefs_hat_room_remote.py"),
    os.path.join(BASE, "gameRooms", "chefs_hat_room_server.py"),
]

replacements = [
    ("from gym.envs.registration import register", "from gymnasium.envs.registration import register"),
    ("from gym import spaces",                     "from gymnasium import spaces"),
    ("import gym\r\n",                             "import gymnasium as gym\r\n"),
    ("import gym\n",                               "import gymnasium as gym\n"),
]

patched = 0
for path in files_to_patch:
    if not os.path.exists(path):
        print(f"  SKIP (not found): {path}")
        continue
    content = open(path, "rb").read().decode("utf-8")
    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
    if changed:
        open(path, "w").write(content)
        print(f"  Patched: {os.path.relpath(path)}")
        patched += 1
    else:
        print(f"  Already clean: {os.path.relpath(path)}")

print(f"\n[Done] {patched} file(s) patched.")
print("[Next] Run: python training/pretrain_vae.py")
