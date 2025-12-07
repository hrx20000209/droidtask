import os
import shutil
from pathlib import Path
from tqdm import tqdm

def copy_states(raw_utgs_root: str, user_tasks_root: str):
    raw_utgs_root = Path(raw_utgs_root)
    user_tasks_root = Path(user_tasks_root)

    # 遍历 raw_utgs 下所有 app
    for app_dir in raw_utgs_root.iterdir():
        if not app_dir.is_dir():
            continue

        states_dir = app_dir / "states"
        if not states_dir.exists():
            print(f"[SKIP] No states folder in {app_dir.name}")
            continue

        # user_tasks/<app>/states/
        target_states_dir = user_tasks_root / app_dir.name / "states"
        target_states_dir.mkdir(parents=True, exist_ok=True)

        screenshot_files = sorted(states_dir.glob("*.png"))

        print(f"\n=== App: {app_dir.name} ===")
        print(f"Found {len(screenshot_files)} screenshots")

        # tqdm 进度条
        for img in tqdm(screenshot_files, desc=f"Copying {app_dir.name}", unit="img"):
            target = target_states_dir / img.name
            shutil.copy(img, target)

        print(f"Copied {len(screenshot_files)} screenshots → {target_states_dir}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    RAW_UTGS = "/Users/huangrunxi/Projects/Dataset/utgs"     # 例如 data/raw_utgs/applauncher/states
    USER_TASKS = "./data/user_tasks" # 例如 data/user_tasks/applauncher/states

    copy_states(RAW_UTGS, USER_TASKS)
