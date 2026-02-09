# Fall Detection & Sentiment Analysis System

A generic, rule-based Python script for detecting falls, assistance needs, and sentiment analysis from operator notes in healthcare alarm monitoring data.

## Overview

This system analyzes operator notes (alarm paths) from healthcare monitoring systems to automatically detect:
- **Falls**: Direct statements, indirect descriptions, injury indicators, emergency responses
- **Assistance**: Physical help provided to members (lift assists, mobility support, etc.)
- **Sentiment**: Emotional tone of the interaction (Negative, Neutral, Positive)

## Features

- ✅ **Generic & Configurable**: Works with any CSV/Parquet file with customizable column names
- ✅ **Rule-Based Detection**: No ML training required - uses pattern matching
- ✅ **Comprehensive Patterns**: 100+ detection patterns based on healthcare domain expertise
- ✅ **Multi-Format Support**: Reads CSV and Parquet, outputs both formats
- ✅ **Easy to Use**: Simple configuration section, clear documentation
- ✅ **Production Ready**: Error handling, validation, detailed logging

## Quick Start

### Prerequisites

```bash
pip install pandas pyarrow
```

### Basic Usage

1. **Update the configuration** in `Standard_Fall_Assist_Sentiment.py`:

```python
CONFIG = {
    'text_column': 'alarm_path',        # Your text column name
    'id_column': 'account_number',      # Your ID column name
    'input_file': 'your_data.parquet',  # Your input file
    'output_parquet': 'output.parquet', # Output parquet file
    'output_csv': 'output.csv',         # Output CSV file
}
```

2. **Run the script**:

```bash
python Standard_Fall_Assist_Sentiment.py
```

## Input Requirements

### Required Columns

Your input file **must** contain these columns:
- **Text Column** (default: `alarm_path`): Operator notes/text to analyze
- **ID Column** (default: `account_number`): Unique member/account identifier

### Optional Columns

Any additional columns will be preserved in the output (e.g., `alarm_id`, `alarm_date`, `alarm_category`, `alarm_resolution`, etc.)

### Supported Formats

- **CSV** (`.csv`, `.CSV`)
- **Parquet** (`.parquet`)

## Output

The script generates 5 new flag columns placed **right after** the text column:

| Column Name | Type | Description |
|------------|------|-------------|
| `fall_count` | int | 1 if fall detected, 0 otherwise |
| `assist_count` | int | 1 if assistance provided, 0 otherwise |
| `sentiment_positive_count` | int | 1 if positive sentiment, 0 otherwise |
| `sentiment_negative_count` | int | 1 if negative sentiment, 0 otherwise |
| `sentiment_neutral_count` | int | 1 if neutral sentiment, 0 otherwise |

**Note**: Only one sentiment flag will be 1 per row (mutually exclusive).

## Detection Logic

### Fall Detection

The system detects falls using multiple signal types with priority handling:

#### 1️⃣ **Exclusions** (Checked First - Highest Priority)
Patterns that should **NOT** be marked as falls:
- Near-falls: "almost fell", "near fall", "caught before fall"
- No fall confirmed: "no fall", "confirmed no fall"
- Prevention/Education: "fall risk assessment", "fall prevention"
- False alarms: "false alarm", "accidental trigger"
- System/Admin: "fall test", "testing fall detection"
- Hypothetical: "in case of fall", "if fall happens"
- Non-fall words: "fall asleep", "fall season"

#### 2️⃣ **Direct Fall Statements** (Strongest Signal)
- "fell", "took a fall", "had a fall"
- "found on floor", "found fallen"
- "witnessed fall", "fall occurred"
- "patient down", "resident down"

#### 3️⃣ **Indirect Descriptions**
- "slipped and fell", "lost balance and fell"
- "tripped and fell", "collapsed to the ground"
- "found lying on floor"

#### 4️⃣ **Injury Indicators** (Linked to Fall)
- "hurt after fall", "injured from fall"
- "bleeding after fall", "pain after fall"

#### 5️⃣ **Emergency Response**
- "EMS dispatched for fall"
- "ambulance called for fall"
- "fall confirmed by EMS"

### Assistance Detection

Detects physical help provided:
- **Standing assistance**: "lift assist", "helped them stand"
- **Floor assistance**: "lifted from floor", "helped off the floor"
- **Wheelchair assistance**: "wheelchair transfer assist"
- **Mobility assistance**: "physical assistance provided", "EMS assisted"

**Exclusions**: "checking in only", "refused assistance", "no assistance given"

### Sentiment Analysis

Analyzes emotional tone with priority: **Negative > Positive > Neutral**

####  **Negative Sentiment** (High Risk Exposure)
- Distress/Pain: "hurt", "pain", "bleeding", "injured"
- Panic/Fear: "panic", "scared", "frightened", "crying"
- Emergency: "emergency", "urgent", "immediate assistance"
- Immobility: "unable to move", "cannot stand"

**Override Rules** (Always force negative):
- Fall + Injury: "fell and hurt", "fall with injury"
- EMS Dispatched: "ambulance called", "911 dispatched"
- Unresponsive: "unconscious", "unresponsive"

####  **Positive Sentiment** (Reassuring/Resolved)
- Calm/Safe: "feeling fine", "okay", "safe", "calm"
- No harm: "no fall", "no injury", "false alarm"
- Resolved: "issue resolved", "situation resolved"
- Gratitude: "thank you", "relieved", "grateful"

#### **Neutral Sentiment** (Informational/Low Emotion)
- Status updates: "reported", "status update"
- Administrative: "device test", "routine check"
- Factual: "requested assistance", "documented"

**Override Rules** (Always force neutral):
- Historical falls: "fall last week", "history of falls"
- Administrative: "device testing", "routine check"

## Configuration Guide

### Basic Configuration

```python
CONFIG = {
    # Column names (REQUIRED - Update these to match your data)
    'text_column': 'alarm_path',        # Column with operator notes
    'id_column': 'account_number',      # Column with member ID
    
    # File paths
    'input_file': 'your_input.parquet',
    'output_parquet': 'output.parquet',
    'output_csv': 'output.csv',
    
    # Output column names (Optional - Customize if needed)
    'fall_flag_column': 'fall_count',
    'assist_flag_column': 'assist_count',
    'sentiment_positive_column': 'sentiment_positive_count',
    'sentiment_negative_column': 'sentiment_negative_count',
    'sentiment_neutral_column': 'sentiment_neutral_count',
}
```

### Advanced: Customize Detection Patterns

You can modify detection patterns in the script:

```python
# Add custom fall patterns
FALL_DIRECT = [
    r'\bfell\b',
    r'\bcustom pattern\b',  # Add your pattern here
]

# Add custom sentiment patterns
SENTIMENT_NEGATIVE = [
    r'\bhurt\b',
    r'\byour custom word\b',  # Add your pattern here
]
```

**Pattern Syntax**:
- `\b` = word boundary (ensures whole word matching)
- `.*` = match any characters
- `|` = OR operator
- Regular expressions supported

## Example Output

### Input Data
```
account_number | alarm_path
---------------|--------------------------------------------------
12345          | Member fell in bathroom, staff assisted, no injury
67890          | False alarm, testing device, no fall occurred
```

### Output Data
```
account_number | alarm_path                                       | fall_count | assist_count | sentiment_positive_count | sentiment_negative_count | sentiment_neutral_count
---------------|--------------------------------------------------|-----------|-------------|-------------------|-------------------|------------------
12345          | Member fell in bathroom, staff assisted...      | 1         | 1           | 0                 | 1                 | 0
67890          | False alarm, testing device, no fall occurred   | 0         | 0           | 1                 | 0                 | 0
```

## 🎓 Use Cases

1. **Quality Assurance**: Automatically flag falls for review
2. **Risk Stratification**: Identify high-risk members based on fall history
3. **Sentiment Monitoring**: Track member distress levels
4. **Operational Analytics**: Measure assistance provided by staff
5. **Compliance**: Document fall incidents for regulatory reporting

## 🛠️ Troubleshooting

### Common Errors

**❌ Error: Column 'alarm_path' not found**
- **Solution**: Update `text_column` in CONFIG to match your column name

**❌ Error: Input file not found**
- **Solution**: Check file path in `input_file` config

**❌ Error: Unsupported file format**
- **Solution**: Ensure file is .csv or .parquet format

### Performance Tips

- **Large files**: Process in chunks or use Parquet format
- **Memory issues**: Reduce file size or increase available RAM
- **Slow processing**: Consider parallel processing for very large datasets

## Validation & Testing

### Sample Statistics Output

```
======================================================================
Summary Statistics
======================================================================
Falls Detected:       1,234 ( 5.67%)
Assistance Detected:  2,456 (11.23%)
Positive Sentiment:   5,678 (25.89%)
Negative Sentiment:   3,456 (15.78%)
Neutral Sentiment:   12,345 (56.34%)
======================================================================
```

### Quality Checks

1. **Validate fall detection**: Review sample of flagged falls
2. **Check sentiment accuracy**: Review sentiment distributions
3. **Verify exclusions**: Ensure false positives are excluded
4. **Test edge cases**: Test with known patterns

##  Contributing

Suggestions for improvement are welcome! Common enhancement areas:
- Additional detection patterns for specific use cases
- New language support
- Performance optimizations
- Additional output formats

## License

This script is provided as-is for healthcare analytics purposes.

## Support

For questions or issues:
1. Check the configuration section
2. Review error messages for guidance
3. Validate your input data format
4. Check pattern matching logic

## Version History

### Version 2.0 (Current)
- ✅ Generic configuration system
- ✅ Support for CSV and Parquet
- ✅ Customizable column names
- ✅ Enhanced documentation
- ✅ Improved error handling

### Version 1.0
- Initial release with fixed column names

## 📚 Additional Resources

### Detection Logic Documentation

For detailed documentation of all detection patterns, see the inline comments in the script or refer to the specification document.

### Pattern Examples

**Fall Detection**:
```
✓ "Patient fell in bathroom"           → fall_count = 1
✓ "Found on floor, assisted up"        → fall_count = 1, assist_count= 1
✗ "Almost fell but caught themselves"  → fall_count = 0
✗ "Fall prevention education provided" → fall_count = 0
```

**Sentiment Analysis**:
```
✓ "Member hurt after fall, EMS called"     → negative
✓ "False alarm, member okay"               → positive
✓ "Status update, routine check"           → neutral

## 🌟 Star This Repository

If you find this tool helpful, please consider giving it a star! ⭐
