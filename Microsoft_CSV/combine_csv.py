import os
import json
import pandas as pd

"""
This script is used to combine three csv files (training, validation, and test sets) from the Microsoft_CSV folder into a single json file named microsoft.json.
1. read the three csv files and concatenate them into a single dataframe.
2. transform the combined dataframe into the specified json structure, which includes extracting the lemma from the
video_id field and adding a source field with the value "microsoft".
3. write the resulting json data to disk with proper error handling and logging.
"""

def read_csv_file(filename):
    """Reads a CSV file and handles potential exceptions."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully read {filename} with {len(df)} records.")
        return df
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return pd.DataFrame()

def combine_datasets(train_path, val_path, test_path):
    """Combines training, validation, and test dataframes."""
    train_df = read_csv_file(train_path)
    val_df = read_csv_file(val_path)
    test_df = read_csv_file(test_path)
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return combined_df

def extract_lemma_from_video_id(video_id_str):
    """
    Extracts the lemma from the video_id format: '[numbers]-[lemma].mp4'
    Example: '12334-seedTOMORROW.MP4' -> 'seedTOMORROW'
    """
    # 1. Clean up any accidental leading/trailing whitespaces
    video_str = str(video_id_str).strip()
    
    # 2. Check for the hyphen separator
    if '-' not in video_str:
        return "unknown"
        
    # 3. Extract the portion after the first hyphen
    after_hyphen = video_str.split('-', 1)[1]
    
    # 4. Safely remove the extension by splitting from the right side at the last dot
    if '.' in after_hyphen:
        lemma = after_hyphen.rsplit('.', 1)[0]
    else:
        lemma = after_hyphen
        
    return lemma if lemma else "unknown"

def transform_to_json_structure(df):
    """Maps the combined dataframe into the specified schema format."""
    json_list = []
    for _, row in df.iterrows():
        # Ensure values are strings and handle potential NaN values safely
        gloss_val = str(row['gloss']).strip() if pd.notna(row['gloss']) else ""
        video_id_val = str(row['video_id']).strip() if pd.notna(row['video_id']) else ""
        
        extracted_lemma = extract_lemma_from_video_id(video_id_val)
        
        json_list.append({
            "gloss": gloss_val,
            "lemma": extracted_lemma,
            "video_id": video_id_val,
            "source": "microsoft"
        })
    return json_list

def main():
    train_file = "Microsoft_CSV\\aslcitizen_training_set.csv"
    val_file = "Microsoft_CSV\\aslcitizen_val_set.csv"
    test_file = "Microsoft_CSV\\aslcitizen_test_set.csv"
    output_filename = "data_preprocessing\\microsoft.json"

    print("--- Starting Data Aggregation Process ---")
    
    combined_df = combine_datasets(train_file, val_file, test_file)
    
    if combined_df.empty:
        print("Error: No data was loaded. Aborting JSON generation.")
        return

    formatted_data = transform_to_json_structure(combined_df)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=4)
        
        print("\n--- Process Completed Successfully ---")
        print(f"Total records combined and structured: {len(formatted_data)}")
        if formatted_data:
            print("Sample entry written:")
            print(json.dumps(formatted_data[0], indent=2))
            
    except IOError as e:
        print(f"Failed to write output JSON payload to disk: {e}")

if __name__ == "__main__":
    main()