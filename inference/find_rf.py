import os


def hunt_the_rf_ghost(directory="."):
    target_string = "FINAL_SCORES: RF:"
    print(f"--- Searching for '{target_string}' in {os.path.abspath(directory)} ---")

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and git folders
        if any(skip in root for skip in ["venv", ".git", "__pycache__", "node_modules"]):
            continue

        for file in files:
            if file.endswith((".py", ".txt", ".json")):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if target_string in line:
                                print(f"FOUND in {path} (Line {line_num}):")
                                print(f"  --> {line.strip()}")
                                print("-" * 40)
                except Exception:
                    continue


if __name__ == "__main__":
    hunt_the_rf_ghost()