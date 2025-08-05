import pandas as pd
import os
import joblib

def get_discontinuation_alerts():
    """
    Loads and displays the pre-built list of accessories recommended
    for discontinuation.
    """
    print("--- Inventory Discontinuation Alerts ---")
    
    # 1. Define file paths
    data_path = os.path.join('models', 'discontinuation_list.joblib')

    # 2. Load the pre-built list
    try:
        discontinuation_list = joblib.load(data_path)
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Discontinuation list not found at '{data_path}'. Please run 'build_discontinuation_list.py' first.")

    # 3. Display the results
    if not discontinuation_list:
        print("\nNo accessories are currently recommended for discontinuation. Inventory looks healthy!")
        return

    print(f"\nFound {len(discontinuation_list)} accessories to put on clearance sale:")
    
    # Convert to DataFrame for pretty printing
    alerts_df = pd.DataFrame(discontinuation_list)
    print(alerts_df.to_string(index=False))


if __name__ == "__main__":
    get_discontinuation_alerts()
