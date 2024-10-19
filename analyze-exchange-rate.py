import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
from pathlib import Path


def get_data_filename():
    # Create a data directory if it doesn't exist
    data_dir = Path('forex_data')
    data_dir.mkdir(exist_ok=True)

    # Create filename with current year and month
    current_date = datetime.now()
    return data_dir / f'eurusd_hourly_{current_date.year}_{current_date.month}.csv'


def is_data_fresh(filepath):
    """Check if the data file is from the current month"""
    if not filepath.exists():
        return False

    # Get file modification time
    file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
    current_time = datetime.now()

    # Consider data fresh if it's from the current month
    return (file_mtime.year == current_time.year and
            file_mtime.month == current_time.month)


def download_forex_data():
    """Download or load cached forex data"""
    data_file = get_data_filename()

    # If we have fresh data, load it from file
    #if is_data_fresh(data_file):
    #    print("Loading cached forex data...")
    #    return pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Otherwise download new data
    print("Downloading new forex data...")
    eurusd = yf.Ticker("EURUSD=X")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    try:
        # Download hourly data for the entire period at once
        print(f"Downloading data from {start_date.date()} to {end_date.date()}")
        data = eurusd.history(start=start_date, end=end_date, interval="1h")

        if data.empty:
            raise Exception("No data downloaded")

        # Save to CSV file
        print(f"Saving data to {data_file}...")
        data.to_csv(data_file)

        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def group_by_trading_day(data):
    # Group hourly data by trading day
    data['Date'] = data.index.date
    daily_groups = dict(tuple(data.groupby(data.index.date)))
    return daily_groups


def calculate_daily_higher_close(daily_groups):
    # Calculate percentage of days where close was higher than open
    count = 0
    total_days = len(daily_groups)

    for date, day_data in daily_groups.items():
        if len(day_data) > 0:
            day_close = day_data['Close'].iloc[-1]
            day_open = day_data['Open'].iloc[0]
            if day_close > day_open:
                count += 1

    return (count / total_days) * 100


def calculate_random_point_vs_close(daily_groups, num_simulations=100):
    results = []

    for sim in range(num_simulations):
        if sim % 300 == 0:  # Progress indicator
            print(f"Running simulation {sim}/{num_simulations}")

        count = 0
        total_days = len(daily_groups)

        for date, day_data in daily_groups.items():
            if len(day_data) > 0:
                # Select a random point in time during the day
                random_idx = np.random.randint(0, len(day_data))
                random_price = day_data['Close'].iloc[random_idx]
                day_close = day_data['Close'].iloc[-1]

                if day_close > random_price:
                    count += 1

        results.append((count / total_days) * 100)

    return np.mean(results)


def calculate_random_points_comparison(daily_groups, num_simulations=100):
    results = []

    for sim in range(num_simulations):
        if sim % 300 == 0:  # Progress indicator
            print(f"Running simulation {sim}/{num_simulations}")

        count = 0
        total_days = len(daily_groups)

        for date, day_data in daily_groups.items():
            if len(day_data) > 1:  # Need at least 2 points
                # Select two random points in time during the day
                idx1, idx2 = np.random.choice(len(day_data), 2, replace=False)
                # Sort indices to ensure temporal order
                idx1, idx2 = sorted([idx1, idx2])

                price1 = day_data['Close'].iloc[idx1]
                price2 = day_data['Close'].iloc[idx2]

                if price2 > price1:
                    count += 1

        results.append((count / total_days) * 100)

    return np.mean(results)


def calculate_threshold_crossing(daily_groups, multiplier, num_simulations=100):
    """
    For each day:
    1. Select a random point in the first half of the day
    2. Calculate threshold = price - (multiplier * overall_stdev)
    3. Check if any subsequent point crosses above the original price
    """

    # First calculate the overall standard deviation
    all_prices = []
    for day_data in daily_groups.values():
        all_prices.extend(day_data['Close'].values)
    overall_stdev = np.std(all_prices)

    print(f"Overall standard deviation: {overall_stdev:.6f}")

    results = []

    for sim in range(num_simulations):
        if sim % 300 == 0:  # Progress indicator
            print(f"Running simulation {sim}/{num_simulations}")

        count = 0
        total_days = len(daily_groups)

        for date, day_data in daily_groups.items():
            if len(day_data) > 1:  # Need at least 2 points
                # Get indices for first half of the day
                mid_point = len(day_data) // 2
                first_half_idx = np.random.randint(0, mid_point)

                # Get the price at the random point
                first_half_price = day_data['Close'].iloc[first_half_idx]

                # Calculate threshold
                threshold = first_half_price + (multiplier * overall_stdev)

                # Check remaining points in the day
                remaining_prices = day_data['Close'].iloc[first_half_idx + 1:]
                #print(f"looking above threshold {threshold} in rem prices {remaining_prices}")

                # Check if any subsequent price is higher than the threshold
                if any(price > threshold for price in remaining_prices):
                    count += 1
                    #print(f"found price bigger than threshold {threshold} with first half price {first_half_price} minus {multiplier*overall_stdev}")
                    #print(f"price list {remaining_prices}")
                #else:
                #    print(f"found point where we don't have that point above threshold. price {price} threshold {threshold}")

        results.append((count / total_days) * 100)

    return np.mean(results), overall_stdev


def run_threshold_analysis(daily_groups, multipliers=[0.01, 0.05,0.1, 0.2, 0.35, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]):
    """Run threshold crossing analysis for different standard deviation multipliers"""
    print("\nRunning Threshold Crossing Analysis...")

    results = {}
    for mult in multipliers:
        print(f"\nAnalyzing with multiplier {mult}...")
        percentage, stdev = calculate_threshold_crossing(daily_groups, mult)
        results[mult] = percentage
        print(f"Multiplier {mult}: {percentage:.2f}% of days crossed the threshold")
        print(f"Threshold calculation: random_price - ({mult} * {stdev:.6f})")

    return results


def main():
    try:
        # Download or load data
        data = download_forex_data()

        # Group data by trading day
        print("Processing data...")
        daily_groups = group_by_trading_day(data)

        # Run threshold crossing analysis
        threshold_results = run_threshold_analysis(daily_groups)

        # Print summary
        print("\nThreshold Crossing Analysis Summary:")
        print("====================================")
        for multiplier, percentage in threshold_results.items():
            print(f"Multiplier {multiplier}: {percentage:.2f}% success rate")

        # Calculate statistics
        #print("\nCalculating statistics...")
        #higher_close_percent = calculate_daily_higher_close(daily_groups)
        #print(f"\nPercentage of days that closed higher than open: {higher_close_percent:.2f}%")

        #random_vs_close = calculate_random_point_vs_close(daily_groups)
        #print(f"\nPercentage of days where close was higher than random point: {random_vs_close:.2f}%")

        #random_points_comparison = calculate_random_points_comparison(daily_groups)
        #print(f"\nPercentage of days where second random point was higher than first: {random_points_comparison:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()