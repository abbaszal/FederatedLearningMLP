import os

data_dir = "HumanGaitDataBase/Data"

bad_files = []
good_files = []

for filename in os.listdir(data_dir):

    if not filename.endswith(".txt"):
        bad_files.append(filename)
        continue

    parts = filename.split('_')

    # Must have at least: ..._<userID>_<trialID>.txt
    if len(parts) < 4:
        bad_files.append(filename)
        continue

    # Extract user ID from second-to-last chunk
    try:
        user_id = int(parts[-2])
    except ValueError:
        bad_files.append(filename)
        continue

    good_files.append(filename)

print("\n=== GOOD FILES ===")
for f in good_files[:10]:
    print("  ✓", f)
print(f"... total good files: {len(good_files)}")

print("\n=== BAD FILES ===")
for f in bad_files:
    print("  ✗", f)
print(f"... total bad files: {len(bad_files)}")
