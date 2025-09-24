# Kaomoji vs Emoticon Analysis Report

## Executive Summary
- **Total Items Analyzed**: 52,124
- **Analysis Date**: 2025-01-20
- **Classification**: Binary (Kaomoji vs Emoticon)

## Results

### Distribution
- **KAOMOJI**: 50,871 items (97.6%)
- **EMOTICON**: 1,253 items (2.4%)

### Top Descriptions by Type

#### KAOMOJI (50,871 items)
- hoa: 9,047 items
- đỏ mặt: 4,046 items
- con gấu: 3,868 items
- trái tim: 3,345 items
- mỉm cười: 3,335 items
- con mèo: 2,506 items
- chó: 2,019 items
- sự tức giận: 1,590 items
- buồn: 1,207 items
- đang chạy: 1,059 items

#### EMOTICON (1,253 items)
- trái tim: 104 items
- ngạc nhiên: 79 items
- mỉm cười: 73 items
- buồn: 70 items
- con gấu: 66 items
- sự tức giận: 58 items
- con mèo: 55 items
- đồ ăn: 47 items
- uốn cong: 46 items
- chỉ: 44 items


## Methodology

### Kaomoji Detection
- Japanese Unicode characters (Hiragana, Katakana, Kanji)
- Complex facial expressions with special symbols
- Decorative elements typical of Japanese emoticons
- Complex structure with multiple special characters

### Emoticon Detection  
- All emoticons not classified as Kaomoji
- Includes Western ASCII emoticons, Vietnamese style, and others

## Files Generated
1. `dataset_kaomoji_emoticon_classified.csv` - Classified dataset
2. `analysis_summary.json` - Structured analysis results
3. `kaomoji_emoticon_*.png` - Visualization charts
4. `detailed_analysis_report.md` - This report
