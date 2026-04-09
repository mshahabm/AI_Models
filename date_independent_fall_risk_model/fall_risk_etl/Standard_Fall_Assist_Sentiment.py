"""
Fall Detection and Sentiment Analysis System
=============================================

A generic rule-based system for detecting falls, assistance needs, and sentiment 
analysis from operator notes in healthcare alarm monitoring data.

Author: Healthcare Analytics Team
Last Updated: January 2026
Version: 2.0 (Generic)
"""

import pandas as pd
import re
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION SECTION - UPDATE THESE SETTINGS FOR YOUR DATA
# ============================================================================

CONFIG = {
    # Required column names in your input file
    'text_column': 'alarm_path',        # Column containing operator notes/text to analyze
    'id_column': 'account_number',      # Column containing unique member/account identifier
    
    # Input/Output file paths
    'input_file': r'C:\Users\ShahabMemon\Documents\Work\fall_risk_etl\fall_risk_etl\data\health\202411_to_202504_operator.parquet',  # Update this to your input file path
    'output_parquet': r'C:\Users\ShahabMemon\Documents\Work\fall_risk_etl\fall_risk_etl\data\health\Fall_Assist_Sentiment_Output.parquet',
    'output_csv': r'C:\Users\ShahabMemon\Documents\Work\fall_risk_etl\fall_risk_etl\data\health\Fall_Assist_Sentiment_Output.csv',
    
    # Output column names (customize if needed)
    'fall_flag_column': 'fall_flag',
    'assist_flag_column': 'assist_flag',
    'sentiment_positive_column': 'sentiment_positive_flag',
    'sentiment_negative_column': 'sentiment_negative_flag',
    'sentiment_neutral_column': 'sentiment_neutral_flag',
}

# ============================================================================
# DETECTION PATTERNS - Edit these if you need to customize detection logic
# ============================================================================

# Fall Detection Patterns
FALL_EXCLUSIONS = [
    # Near-fall / Almost-fall
    r'\balmost fell\b', r'\bnear fall\b', r'\bnearly fell\b', r'\bcaught before fall\b',
    # No fall confirmed
    r'\bno fall\b', r'\bno fall occurred\b', r'\bconfirmed no fall\b',
    # Prevention/Education
    r'\bfall risk assessment\b', r'\bfall prevention\b', r'\bdiscussed fall risk\b',
    # False alarms
    r'\bfalse alarm\b', r'\bfalse fall alarm\b', r'\baccidental trigger\b',
    # System/Admin
    r'\bfall test\b', r'\btest fall\b', r'\btesting fall detection\b',
    # Hypothetical
    r'\bin case of fall\b', r'\bif fall happens\b', r'\bif you fall\b',
    # Non-fall words
    r'\bfall asleep\b', r'\bfell asleep\b', r'\bfall season\b',
]

FALL_DIRECT = [
    r'\bfell\b', r'\btook a fall\b', r'\bhad a fall\b', r'\bfall occurred\b',
    r'\bfound on floor\b', r'\bfound fallen\b', r'\bwitnessed fall\b',
    r'\bpatient down\b', r'\bresident down\b',
]

FALL_INDIRECT = [
    r'\bslipped and fell\b', r'\blost balance and fell\b', r'\btripped and fell\b',
    r'\bcollapsed to the ground\b', r'\bfound lying on floor\b',
    r'\bunable to get up after fall\b', r'\blift assist after fall\b',
]

FALL_INJURY = [
    r'\bhurt after fall\b', r'\binjured from fall\b', r'\bbleeding after fall\b',
    r'\bpain after fall\b', r'\bhead injury after fall\b',
]

FALL_EMERGENCY = [
    r'\bems dispatched for fall\b', r'\bambulance called for fall\b',
    r'\bfall confirmed by ems\b', r'\bems responded to fall\b',
]

# Assistance Detection Patterns
ASSIST_EXCLUSIONS = [
    r'\bchecking in only\b', r'\baccidental press\b', r'\bno assistance given\b',
    r'\brefused assistance\b', r'\bdeclined assistance\b',
]

ASSIST_PATTERNS = [
    r'\blift assist\b', r'\bhelped them stand\b', r'\bassisted to stand\b',
    r'\bhelped up\b', r'\blifted from floor\b', r'\bhelped off the floor\b',
    r'\bwheelchair transfer assist\b', r'\bphysical assistance provided\b',
    r'\bems assisted\b', r'\bfire dept assisted\b',
]

# Sentiment Patterns
SENTIMENT_NEGATIVE = [
    r'\bfell\b', r'\bfall\b', r'\bhurt\b', r'\bpain\b', r'\bbleeding\b', r'\binjured\b',
    r'\bpanic\b', r'\bscared\b', r'\bemergency\b', r'\bunable to move\b',
]

SENTIMENT_POSITIVE = [
    r'\bfeeling fine\b', r'\bokay\b', r'\bsafe\b', r'\bresolved\b', r'\bno fall\b',
    r'\bno injury\b', r'\bthank you\b', r'\brelieved\b',
]

SENTIMENT_NEUTRAL = [
    r'\breported\b', r'\bstatus update\b', r'\bdevice test\b',
    r'\brequested assistance\b', r'\bdocumented\b',
]

# Override patterns (force negative sentiment)
OVERRIDE_NEGATIVE = [
    # Fall + Injury
    (r'\bfall.*injur', r'\binjur.*fall', r'\bfell.*hurt', r'\bfall.*pain'),
    # EMS dispatched
    (r'\bems dispatched\b', r'\bambulance called\b', r'\b911 called\b'),
    # Unable to respond
    (r'\bunresponsive\b', r'\bunconscious\b', r'\bunable to move\b'),
]

# Override patterns (force neutral sentiment)
OVERRIDE_NEUTRAL = [
    # Historical falls
    (r'\bfall last week\b', r'\bfall yesterday\b', r'\bhistory of falls\b'),
    # Administrative
    (r'\bdevice testing\b', r'\btest call\b', r'\broutine check\b'),
]

# ============================================================================
# CORE DETECTION FUNCTIONS
# ============================================================================

def detect_fall(text):
    """Detect if a fall occurred. Returns 1 if fall detected, 0 otherwise."""
    if pd.isna(text) or text == '':
        return 0
    
    text_lower = str(text).lower()
    
    # Check exclusions first
    if any(re.search(pattern, text_lower) for pattern in FALL_EXCLUSIONS):
        return 0
    
    # Special handling: "on the floor"
    if re.search(r'\bon the floor\b', text_lower):
        if not re.search(r'\b(assisted|lowered) to the floor\b', text_lower):
            return 1
    
    # Check all fall patterns
    all_patterns = FALL_DIRECT + FALL_INDIRECT + FALL_INJURY + FALL_EMERGENCY
    if any(re.search(pattern, text_lower) for pattern in all_patterns):
        return 1
    
    return 0


def detect_assistance(text):
    """Detect if assistance was provided/needed. Returns 1 if yes, 0 otherwise."""
    if pd.isna(text) or text == '':
        return 0
    
    text_lower = str(text).lower()
    
    # Check exclusions first
    if any(re.search(pattern, text_lower) for pattern in ASSIST_EXCLUSIONS):
        return 0
    
    # Check assistance patterns
    if any(re.search(pattern, text_lower) for pattern in ASSIST_PATTERNS):
        return 1
    
    return 0


def analyze_sentiment(text, fall_flag=0):
    """
    Analyze sentiment. Returns tuple: (positive_flag, negative_flag, neutral_flag)
    Priority: Negative > Positive > Neutral
    """
    if pd.isna(text) or text == '':
        return (0, 0, 1)  # Default to neutral
    
    text_lower = str(text).lower()
    
    # Check override patterns first (highest priority)
    for override_group in OVERRIDE_NEGATIVE:
        if any(re.search(pattern, text_lower) for pattern in override_group):
            return (0, 1, 0)  # Force NEGATIVE
    
    for override_group in OVERRIDE_NEUTRAL:
        if any(re.search(pattern, text_lower) for pattern in override_group):
            return (0, 0, 1)  # Force NEUTRAL
    
    # Count sentiment indicators
    neg_score = sum(1 for p in SENTIMENT_NEGATIVE if re.search(p, text_lower))
    pos_score = sum(1 for p in SENTIMENT_POSITIVE if re.search(p, text_lower))
    neu_score = sum(1 for p in SENTIMENT_NEUTRAL if re.search(p, text_lower))
    
    # Apply priority logic
    if neg_score > 0:
        return (0, 1, 0)  # Negative
    if pos_score > 0:
        return (1, 0, 0)  # Positive
    if neu_score > 0:
        return (0, 0, 1)  # Neutral
    
    return (0, 0, 1)  # Default neutral


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_file(input_file, config=CONFIG):
    """
    Process input file and generate fall/assist/sentiment flags.
    
    Parameters:
    -----------
    input_file : str
        Path to input file (CSV or Parquet)
    config : dict
        Configuration dictionary with column names and settings
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with new flag columns
    """
    print(f"\n{'='*70}")
    print(f"Fall Detection & Sentiment Analysis System")
    print(f"{'='*70}")
    print(f"Input file: {input_file}")
    
    # Read input file (auto-detect format)
    file_ext = Path(input_file).suffix.lower()
    if file_ext == '.parquet':
        df = pd.read_parquet(input_file)
    elif file_ext in ['.csv', '.CSV']:
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .parquet or .csv")
    
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # Validate required columns
    text_col = config['text_column']
    id_col = config['id_column']
    
    if text_col not in df.columns:
        raise ValueError(f"Required column '{text_col}' not found. Available: {list(df.columns)}")
    if id_col not in df.columns:
        raise ValueError(f"Required column '{id_col}' not found. Available: {list(df.columns)}")
    
    print(f"Text column: '{text_col}'")
    print(f"ID column: '{id_col}'")
    
    # Process detection
    print(f"\n{'='*70}")
    print("Processing...")
    print(f"{'='*70}")
    
    print("→ Detecting falls...")
    df[config['fall_flag_column']] = df[text_col].apply(detect_fall)
    
    print("→ Detecting assistance...")
    df[config['assist_flag_column']] = df[text_col].apply(detect_assistance)
    
    print("→ Analyzing sentiment...")
    sentiment_results = df.apply(
        lambda row: analyze_sentiment(row[text_col], row[config['fall_flag_column']]), 
        axis=1
    )
    df[config['sentiment_positive_column']] = sentiment_results.apply(lambda x: x[0])
    df[config['sentiment_negative_column']] = sentiment_results.apply(lambda x: x[1])
    df[config['sentiment_neutral_column']] = sentiment_results.apply(lambda x: x[2])
    
    # Reorder columns: place new columns after text column
    text_col_idx = df.columns.get_loc(text_col)
    new_cols = [
        config['fall_flag_column'],
        config['assist_flag_column'],
        config['sentiment_positive_column'],
        config['sentiment_negative_column'],
        config['sentiment_neutral_column'],
    ]
    
    cols = [c for c in df.columns if c not in new_cols]
    for i, col in enumerate(new_cols):
        cols.insert(text_col_idx + 1 + i, col)
    
    df = df[cols]
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print(f"Falls Detected:       {df[config['fall_flag_column']].sum():6,} ({df[config['fall_flag_column']].mean()*100:5.2f}%)")
    print(f"Assistance Detected:  {df[config['assist_flag_column']].sum():6,} ({df[config['assist_flag_column']].mean()*100:5.2f}%)")
    print(f"Positive Sentiment:   {df[config['sentiment_positive_column']].sum():6,} ({df[config['sentiment_positive_column']].mean()*100:5.2f}%)")
    print(f"Negative Sentiment:   {df[config['sentiment_negative_column']].sum():6,} ({df[config['sentiment_negative_column']].mean()*100:5.2f}%)")
    print(f"Neutral Sentiment:    {df[config['sentiment_neutral_column']].sum():6,} ({df[config['sentiment_neutral_column']].mean()*100:5.2f}%)")
    print(f"{'='*70}")
    
    return df


def save_output(df, config=CONFIG):
    """Save processed dataframe to output files."""
    print(f"\nSaving outputs...")
    
    # Save Parquet
    if config['output_parquet']:
        df.to_parquet(config['output_parquet'], index=False)
        size_mb = os.path.getsize(config['output_parquet']) / (1024 * 1024)
        print(f"✓ Parquet: {config['output_parquet']} ({size_mb:.2f} MB)")
    
    # Save CSV
    if config['output_csv']:
        df.to_csv(config['output_csv'], index=False)
        size_mb = os.path.getsize(config['output_csv']) / (1024 * 1024)
        print(f"✓ CSV:     {config['output_csv']} ({size_mb:.2f} MB)")
    
    print(f"\n{'='*70}")
    print("✓ Processing Complete!")
    print(f"{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Check if input file exists
        if not os.path.exists(CONFIG['input_file']):
            print(f"\n❌ Error: Input file '{CONFIG['input_file']}' not found!")
            print(f"\nPlease update the 'input_file' in the CONFIG section.")
            exit(1)
        
        # Process the file
        df = process_file(CONFIG['input_file'], CONFIG)
        
        # Save outputs
        save_output(df, CONFIG)
        
        print(f"Successfully processed {len(df):,} rows!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
