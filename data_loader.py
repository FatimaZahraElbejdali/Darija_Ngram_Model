import os


def load_corpus(data_folder="music-data", max_files=None):
    if not os.path.isdir(data_folder):
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    texts = []
    count = 0

    for root, dirs, files in os.walk(data_folder):
        dirs.sort()

        for file in sorted(files):
            if file.startswith("."):
                continue

            path = os.path.join(root, file)

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
                    count += 1

                if max_files is not None and count >= max_files:
                    break

            except OSError as exc:
                print(f"Skipping unreadable file {path}: {exc}")

        if max_files is not None and count >= max_files:
            break

    return "\n".join(texts)
