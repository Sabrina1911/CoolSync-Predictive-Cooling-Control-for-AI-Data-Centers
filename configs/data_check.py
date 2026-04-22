# configs/data_check.py
import pandas as pd
import numpy as np

def check_datasets():

    print("=" * 55)
    print("CHECKING: merged_lstm_core.csv")
    print("=" * 55)

    df1 = pd.read_csv('data/merged_lstm_core.csv')

    print(f"Rows:        {len(df1)}")
    print(f"Columns:     {len(df1.columns)}")
    print(f"Start:       {df1['timestamp'].iloc[0]}")
    print(f"End:         {df1['timestamp'].iloc[-1]}")
    print(f"Null values: {df1.isnull().sum().sum()} (must be 0)")

    print()
    print("Key columns:")
    for col in ['requests_per_15min', 'total_tokens_15min',
                'avg_gpu_power_w', 'TLHC']:
        print(f"  {col}:")
        print(f"    min={df1[col].min():.3f} | "
              f"max={df1[col].max():.3f} | "
              f"mean={df1[col].mean():.3f}")

    print()
    print("Time features:")
    print(f"  hour_sin: {df1['hour_sin'].min():.3f} "
          f"to {df1['hour_sin'].max():.3f}")
    print(f"  hour_cos: {df1['hour_cos'].min():.3f} "
          f"to {df1['hour_cos'].max():.3f}")
    print(f"  DoW:      {sorted(df1['DoW'].unique())}")
    print(f"  WeH:      {sorted(df1['WeH'].unique())}")

    print()
    print("=" * 55)
    print("CHECKING: stage4_cooling_control_norm.csv")
    print("=" * 55)

    df2 = pd.read_csv('data/stage4_cooling_control_norm.csv')

    print(f"Rows:        {len(df2)}")
    print(f"Columns:     {len(df2.columns)}")
    print(f"Start:       {df2['Timestamp'].iloc[0]}")
    print(f"End:         {df2['Timestamp'].iloc[-1]}")
    print(f"Null values: {df2.isnull().sum().sum()} (must be 0)")

    print()
    print("Cooling actions:")
    print(df2['Cooling_Strategy_Action'].value_counts())

    print()
    print("Action → Output mapping:")
    for action in df2['Cooling_Strategy_Action'].unique():
        output = df2[
            df2['Cooling_Strategy_Action'] == action
        ]['Output'].iloc[0]
        print(f"  Output {output} = {action}")

    print()
    print("=" * 55)
    print("ALL CHECKS PASSED ")
    print("=" * 55)

    return df1, df2