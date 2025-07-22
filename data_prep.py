import pandas as pd

def has_easy_parking(parking_str):
    if not isinstance(parking_str, str):
        return 0
    parking_lower = parking_str.lower()
    return int("garage parking" in parking_lower or "validated parking" in parking_lower)

def load_and_clean_data(filepath):
    # Load raw data
    df = pd.read_excel(filepath)

    # Binary columns where blanks mean no / 0
    binary_cols = [
        'vegan_options', 'wheelchair_access', 'outdoor_seating', 'dogs_allowed',
        'kid_friendly', 'gender_neutral_restrooms', 'good_for_working', 'good_for_breakfast',
        'good_for_brunch', 'good_for_lunch', 'offers_catering', 'covered_outdoor_seating',
        'plastic_free_packaging', 'compostable_containers', 'reusable_tableware',
        'tipping_optional', 'accepts_android_pay', 'accepts_apple_pay',
        'accepts_credit_cards', 'military_discount', 'open_to_all',
    ]

    # Fill missing binary cols with 0 and convert to int
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Add easy_parking feature
    if 'parking' in df.columns:
        df['easy_parking'] = df['parking'].apply(has_easy_parking)
    else:
        df['easy_parking'] = 0

    return df

if __name__ == "__main__":
    filepath = "data/coffeeshop_data.xlsx"
    cleaned_df = load_and_clean_data(filepath)
    print(cleaned_df.head())
    cleaned_df.to_csv("data/coffeeshop_data_cleaned.csv", index=False)