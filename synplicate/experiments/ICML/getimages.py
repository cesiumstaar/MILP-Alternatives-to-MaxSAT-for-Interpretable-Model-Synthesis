import shutil
import os
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION ---
# True: Only prints what will happen (Safe mode)
# False: Actually moves files and deletes folders
DRY_RUN = False
# ---------------------

def get_creation_time(path):
    """
    Returns the creation time formatted for use in a filename.
    Format: YYYY-MM-DD_HH-MM-SS
    """
    try:
        # Try to get creation time (stat().st_birthtime is for MacOS/BSD)
        timestamp = path.stat().st_birthtime
    except AttributeError:
        # Fallback to modification time (Linux usually doesn't store birthtime)
        timestamp = path.stat().st_mtime
    
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')

def process_folders():
    base_dir = Path('.')
    print(f"Scanning directory: {base_dir.resolve()}\n")

    # 1. Iterate over Dataset folders (e.g., Adult3)
    for dataset_folder in base_dir.iterdir():
        if not dataset_folder.is_dir():
            continue

        previous_runs_path = dataset_folder / 'previous_runs'
        
        if not previous_runs_path.exists():
            continue

        print(f"Processing: {previous_runs_path}")

        # 2. Iterate over contents of 'previous_runs'
        for item in previous_runs_path.iterdir():
            
            # Only process folders (ignore existing independent images)
            if item.is_dir():
                run_folder = item
                original_folder_name = run_folder.name
                
                # --- NAMING LOGIC ---
                # Split the folder name from the right at the first underscore
                if '_' in original_folder_name:
                    base_name, _ = original_folder_name.rsplit('_', 1)
                else:
                    # Fallback if no underscore exists
                    base_name = original_folder_name

                # Get creation time of the folder
                creation_time = get_creation_time(run_folder)
                
                # Construct new name: base_name + creation_time + extension
                new_filename = f"{base_name}_{creation_time}.png"
                # --------------------

                destination = previous_runs_path / new_filename
                
                # Locate the single PNG inside
                png_files = list(run_folder.glob('*.png'))

                if len(png_files) == 1:
                    target_image = png_files[0]

                    if DRY_RUN:
                        print(f"  [DRY RUN] Rename logic: '{original_folder_name}' -> '{new_filename}'")
                        print(f"            Action: Move contents out -> Delete folder")
                    else:
                        try:
                            # Move and rename the image
                            shutil.move(str(target_image), str(destination))
                            
                            # Delete the old folder and its remaining contents
                            shutil.rmtree(run_folder)
                            print(f"  [SUCCESS] Created: {new_filename}")
                        except Exception as e:
                            print(f"  [ERROR] Failed on {original_folder_name}: {e}")
                
                elif len(png_files) == 0:
                    print(f"  [SKIP] No .png file found inside {original_folder_name}")
                else:
                    print(f"  [SKIP] Multiple .png files inside {original_folder_name}, skipping to be safe.")

        print("-" * 30)

if __name__ == "__main__":
    process_folders()
    if DRY_RUN:
        print("\n*** DRY RUN COMPLETE ***")
        print("Set 'DRY_RUN = False' inside the script to execute changes.")