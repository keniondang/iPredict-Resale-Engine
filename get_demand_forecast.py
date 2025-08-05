import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def get_demand_forecast(category_value: str, periods: int = 90):
    """
    Loads a pre-trained Prophet model with logistic growth parameters
    and generates a forecast for a specified number of future periods.

    Args:
        category_value (str): The category value of the model to load (e.g., 'iPhone 14').
        periods (int): The number of days to forecast into the future.
    """
    print(f"--- Generating Demand Forecast for Category: {category_value} ---")

    # 1. Define file paths
    model_folder = 'models'
    safe_category_value = "".join(c for c in category_value if c.isalnum())
    model_path = os.path.join(model_folder, f'demand_forecast_{safe_category_value}.joblib')

    # 2. Load the pre-trained model package
    print(f"Step 1: Loading model package from '{model_path}'...")
    try:
        model_package = joblib.load(model_path)
        model = model_package['model']
        release_date = model_package['release_date']
        successor_date = model_package['successor_date']
        base_cap = model_package['base_cap']
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Model not found. Please run 'train_demand_forecast.py' for category '{category_value}' first.")
    except KeyError:
        raise SystemExit("ERROR: Model file is outdated. Please re-run 'train_demand_forecast.py' to save the model with logistic growth parameters.")


    # 3. Create a dataframe for future dates and calculate the future cap
    future_df = model.make_future_dataframe(periods=periods)
    
    # Define the same cap calculation function used during training
    def calculate_cap(row_date):
        days_since_release = (row_date - release_date).days
        decay_factor = 1 - (days_since_release / (365 * 4))
        
        if pd.notna(successor_date) and row_date > successor_date:
            days_after_successor = (row_date - successor_date).days
            successor_decay = 0.5 - (days_after_successor / (365 * 2))
            decay_factor = min(decay_factor, successor_decay)

        return max(1, base_cap * decay_factor)

    # Apply the cap and floor to the future dataframe
    future_df['cap'] = future_df['ds'].apply(calculate_cap)
    future_df['floor'] = 0

    # 4. Generate the forecast
    print(f"Step 2: Generating forecast for the next {periods} days...")
    forecast_df = model.predict(future_df)

    # 5. Display the forecast results
    print("\n--- Forecast Results ---")
    print("Forecasted sales for the upcoming days:")
    print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'cap']].tail())
    
    total_forecast = forecast_df['yhat'][-periods:].sum()
    print(f"\nEstimated total demand for the next {periods} days: {int(round(total_forecast))} units")

    # 6. Visualize the forecast
    print("\nStep 3: Generating forecast plots (a window may open)...")
    
    fig1 = model.plot(forecast_df)
    plt.title(f"Sales Forecast for '{category_value}' Category (with Decaying Cap)")
    plt.xlabel("Date")
    plt.ylabel("Daily Sales")
    plt.show()

    fig2 = model.plot_components(forecast_df)
    plt.suptitle(f"Forecast Components for '{category_value}' Category")
    plt.show()


if __name__ == "__main__":
    # --- Example: Get a forecast for an older model to see the decay ---
    # This value must match one of the models you trained
    # For example: 'iPhone 12', 'iPhone 13 Pro', etc.
    get_demand_forecast(category_value='iPhone 16 Pro 512GB', periods=365)
