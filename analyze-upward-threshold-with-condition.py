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


def download_forex_data():
    """Download or load cached forex data"""
    data_file = get_data_filename()

    # download new data
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

    good_trade_count = 0
    bad_trade_count = 0
    no_action_count = 0
    for sim in range(num_simulations):
        if sim % 300 == 0:  # Progress indicator
            print(f"Running simulation {sim}/{num_simulations}")

        count = 0
        # if random val is higher than start of day, don't trade.
        total_days = len(daily_groups)

        for date, day_data in daily_groups.items():
            if len(day_data) > 1:  # Need at least 2 points
                # Get indices for first half of the day
                mid_point = len(day_data) // 2
                first_half_idx = np.random.randint(0, mid_point)

                # Get the price at the random point
                first_half_price = day_data['Close'].iloc[first_half_idx]
                start_day_price = day_data['Close'].iloc[0]

                # Calculate threshold
                threshold = first_half_price + (multiplier * overall_stdev)

                # Check remaining points in the day
                remaining_prices = day_data['Close'].iloc[first_half_idx + 1:]
                # print(f"looking above threshold {threshold} in rem prices {remaining_prices}")

                # original trading, no condition of below start.
                if any(price > threshold for price in remaining_prices):
                    count += 1

                # trade only if price lower than start
                if first_half_price > start_day_price: # don't trade, too high
                    no_action_count += 1
                # only if according to condition of trading (low price)
                # Check if any subsequent price is higher than the threshold
                elif any(price > threshold for price in remaining_prices):
                    good_trade_count += 1
                else:
                    bad_trade_count += 1
                    # print(f"found price bigger than threshold {threshold} with first half price {first_half_price} minus {multiplier*overall_stdev}")
                    # print(f"price list {remaining_prices}")
                # else:
                #    print(f"found point where we don't have that point above threshold. price {price} threshold {threshold}")

        results.append((count / total_days) * 100)

    #return np.mean(results), overall_stdev
    good_pct = good_trade_count / (good_trade_count + bad_trade_count) * 100
    print(f"done simulation: good trades {good_trade_count} bad trades {bad_trade_count} skipped {no_action_count} pct {good_pct} pct for all {np.mean(results)}")
    return good_pct, overall_stdev


def run_threshold_analysis(daily_groups, multipliers=[0.01, 0.05, 0.1, 0.2]):
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

        # select random point in the day, check if it is possible to sell at
        # profit of e.g. 5% of stdev, calculate % of times it's possible.
        # Run threshold crossing analysis
        threshold_results = run_threshold_analysis(daily_groups)

        # Print summary
        print("\nThreshold Crossing Analysis Summary:")
        print("====================================")
        for multiplier, percentage in threshold_results.items():
            print(f"Multiplier {multiplier}: {percentage:.2f}% success rate")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()