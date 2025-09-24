#!/usr/bin/env python3
"""
Kaomoji vs Emoticon Analysis
Phân tích dataset chỉ 2 loại: Kaomoji (Nhật Bản) và Emoticon (các loại khác)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from typing import Dict, List
from pathlib import Path


class KaomojiEmoticonAnalyzer:
    def __init__(self):
        self.setup_kaomoji_patterns()
    
    def setup_kaomoji_patterns(self):
        """Setup patterns để detect Kaomoji (Japanese style)"""
        
        # Kaomoji indicators - đặc trưng của emoticon Nhật Bản
        self.kaomoji_indicators = [
            # Japanese/Unicode characters
            r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',  # Hiragana, Katakana, Kanji
            # Special Japanese face characters
            r'[ಠಥ◕⌒´`~‾°º○●◎⊙＊✧★☆ツシωε]',
            # Complex Japanese parentheses
            r'[（）]',
            # Box drawing và special symbols phổ biến trong Kaomoji
            r'[╯╰╭╮╱╲╳╴╵╶╷┬─┳┷┻━┃┏┓┗┛├┤┼]',
            # Arrows và geometric shapes trong Kaomoji
            r'[◄►▶◀▲△▼▽◊◈◇◆■□▪▫]',
            # Face elements đặc trưng Kaomoji
            r'[ᵕᴗᴖᵔᴥεッㅂㅇㅅㅁㅜㅠ]',
            # Kaomoji decorative elements
            r'[≡━═┈┉┅┄⋯…‥∴∵⌒＾∀∩∪⊂⊃⊆⊇]',
        ]
        
        # Complex structure patterns đặc trưng Kaomoji
        self.kaomoji_structure_patterns = [
            r'[（\(][^)]{4,}[）\)]',  # Complex content trong ngoặc
            r'[^a-zA-Z0-9\s]{4,}',   # Chuỗi dài toàn special characters
            r'[\u3000-\u303F]',      # Japanese punctuation và symbols
            r'[\uFF00-\uFFEF]',      # Full-width characters
        ]
    
    def is_kaomoji(self, emoticon: str) -> bool:
        """Kiểm tra xem có phải Kaomoji (Japanese style) không"""
        emoticon = emoticon.strip()
        
        if not emoticon:
            return False
        
        # Đếm các indicators của Kaomoji
        indicator_count = 0
        for pattern in self.kaomoji_indicators:
            if re.search(pattern, emoticon):
                indicator_count += 1
        
        # Nếu có nhiều indicators → Kaomoji
        if indicator_count >= 2:
            return True
        
        # Có 1 indicator + cấu trúc phức tạp → Kaomoji  
        if indicator_count >= 1:
            # Kiểm tra độ phức tạp
            if len(emoticon) > 4:  # Dài
                return True
            # Có non-ASCII characters
            if re.search(r'[^\x00-\x7F]', emoticon):
                return True
        
        # Kiểm tra cấu trúc đặc trưng Kaomoji
        for pattern in self.kaomoji_structure_patterns:
            if re.search(pattern, emoticon):
                return True
        
        # Emoticon rất dài và phức tạp → có thể là Kaomoji
        if len(emoticon) > 8:
            return True
            
        return False
    
    def classify_emoticon(self, emoticon: str) -> str:
        """Phân loại emoticon: kaomoji hoặc emoticon"""
        if self.is_kaomoji(emoticon):
            return 'kaomoji'
        else:
            return 'emoticon'
    
    def analyze_dataset(self, csv_file: str) -> Dict:
        """Phân tích dataset và tạo visualizations"""
        
        print("🎌 KAOMOJI vs EMOTICON ANALYSIS")
        print("=" * 60)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        print(f"📊 Total entries: {len(df):,}")
        
        # Classify từng emoticon
        print("🔄 Classifying emoticons...")
        df['type'] = df['emoticon_code'].apply(self.classify_emoticon)
        
        # Count by type
        type_counts = df['type'].value_counts()
        total = len(df)
        
        print("\n📈 TYPE DISTRIBUTION:")
        for etype, count in type_counts.items():
            percentage = (count / total) * 100
            print(f"   {etype.upper():<12}: {count:>7,} items ({percentage:5.1f}%)")
        
        # Analyze descriptions
        self.analyze_descriptions(df)
        
        # Create visualizations
        self.create_visualizations(type_counts, total)
        
        # Show examples
        self.show_examples(df)
        
        # Save results
        self.save_results(df, type_counts, total)
        
        return {
            'total': total,
            'type_counts': type_counts.to_dict(),
            'percentages': {k: (v/total)*100 for k, v in type_counts.to_dict().items()},
            'dataframe': df
        }
    
    def analyze_descriptions(self, df: pd.DataFrame):
        """Phân tích descriptions theo type"""
        
        print("\n📝 TOP DESCRIPTIONS BY TYPE:")
        
        for etype in ['kaomoji', 'emoticon']:
            subset = df[df['type'] == etype]
            if len(subset) > 0:
                top_descriptions = subset['description'].value_counts().head(10)
                print(f"\n{etype.upper()} ({len(subset):,} total):")
                for desc, count in top_descriptions.items():
                    print(f"   • {desc}: {count:,} items")
    
    def create_visualizations(self, type_counts: pd.Series, total: int):
        """Tạo các biểu đồ phân tích"""
        
        # Setup matplotlib
        plt.style.use('seaborn-v0_8')
        
        # 1. Pie Chart
        self.create_pie_chart(type_counts, total)
        
        # 2. Bar Chart  
        self.create_bar_chart(type_counts)
        
        # 3. Donut Chart
        self.create_donut_chart(type_counts, total)
    
    def create_pie_chart(self, type_counts: pd.Series, total: int):
        """Tạo pie chart"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colors cho từng type
        colors = ['#FF6B6B', '#4ECDC4']  # Red cho Kaomoji, Teal cho Emoticon
        
        types = type_counts.index.tolist()
        counts = type_counts.values.tolist()
        percentages = [(count/total)*100 for count in counts]
        
        # Create pie
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f'{t.upper()}\n{p:.1f}%' for t, p in zip(types, percentages)],
            colors=colors,
            autopct='%1.0f',
            startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'},
            wedgeprops={'linewidth': 3, 'edgecolor': 'white'}
        )
        
        ax.set_title('🎌 Kaomoji vs Emoticon Distribution\n📊 Total: {:,} Items'.format(total), 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('visualize/kaomoji_emoticon_pie_chart.png', dpi=300, bbox_inches='tight')
        print("📊 Pie chart saved: visualize/kaomoji_emoticon_pie_chart.png")
        plt.close()
    
    def create_bar_chart(self, type_counts: pd.Series):
        """Tạo bar chart"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#FF6B6B', '#4ECDC4']
        bars = ax.bar(type_counts.index, type_counts.values, color=colors)
        
        # Add value labels
        for bar, count in zip(bars, type_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title('📊 Kaomoji vs Emoticon Distribution (Bar Chart)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Type', fontweight='bold')
        ax.set_ylabel('Number of Items', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualize/kaomoji_emoticon_bar_chart.png', dpi=300, bbox_inches='tight')
        print("📊 Bar chart saved: visualize/kaomoji_emoticon_bar_chart.png")
        plt.close()
    
    def create_donut_chart(self, type_counts: pd.Series, total: int):
        """Tạo donut chart"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4']
        types = type_counts.index.tolist()
        counts = type_counts.values.tolist()
        percentages = [(count/total)*100 for count in counts]
        
        # Create donut
        wedges, texts = ax.pie(counts, colors=colors, startangle=90, 
                              wedgeprops=dict(width=0.5, linewidth=3, edgecolor='white'))
        
        # Add center text
        ax.text(0, 0, f'Total\n{total:,}\nItems', ha='center', va='center', 
                fontsize=16, fontweight='bold')
        
        # Add legend
        legend_labels = [f'{t.upper()}: {c:,} ({p:.1f}%)' 
                        for t, c, p in zip(types, counts, percentages)]
        ax.legend(wedges, legend_labels, title="Types", loc="center left", 
                 bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_title('🎌 Kaomoji vs Emoticon (Donut Chart)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('visualize/kaomoji_emoticon_donut_chart.png', dpi=300, bbox_inches='tight')
        print("📊 Donut chart saved: visualize/kaomoji_emoticon_donut_chart.png")
        plt.close()
    
    def show_examples(self, df: pd.DataFrame):
        """Hiển thị examples của từng loại"""
        
        print("\n📝 EXAMPLES BY TYPE:")
        
        for etype in ['kaomoji', 'emoticon']:
            subset = df[df['type'] == etype]
            if len(subset) > 0:
                examples = subset['emoticon_code'].head(20).tolist()
                print(f"\n{etype.upper()} ({len(subset):,} total) - Examples:")
                for i, example in enumerate(examples, 1):
                    print(f"   {i:2}. {example}")
    
    def save_results(self, df: pd.DataFrame, type_counts: pd.Series, total: int):
        """Lưu kết quả analysis"""
        
        # 1. Save classified dataset
        output_file = 'visualize/dataset_kaomoji_emoticon_classified.csv'
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Classified dataset saved: {output_file}")
        
        # 2. Save analysis summary
        summary = {
            'analysis_date': '2025-01-20',
            'total_items': int(total),
            'type_distribution': {
                k: {
                    'count': int(v),
                    'percentage': round((v/total)*100, 2)
                }
                for k, v in type_counts.to_dict().items()
            },
            'methodology': {
                'kaomoji_detection': 'Japanese Unicode characters, complex structures, special symbols',
                'emoticon_detection': 'All other emoticons not classified as Kaomoji'
            }
        }
        
        with open('visualize/analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("📋 Analysis summary saved: visualize/analysis_summary.json")
        
        # 3. Save detailed report
        self.create_detailed_report(df, type_counts, total)
    
    def create_detailed_report(self, df: pd.DataFrame, type_counts: pd.Series, total: int):
        """Tạo báo cáo chi tiết"""
        
        report = f"""# Kaomoji vs Emoticon Analysis Report

## Executive Summary
- **Total Items Analyzed**: {total:,}
- **Analysis Date**: 2025-01-20
- **Classification**: Binary (Kaomoji vs Emoticon)

## Results

### Distribution
"""
        
        for etype, count in type_counts.items():
            percentage = (count / total) * 100
            report += f"- **{etype.upper()}**: {count:,} items ({percentage:.1f}%)\n"
        
        report += f"""
### Top Descriptions by Type

"""
        
        # Add top descriptions for each type
        for etype in ['kaomoji', 'emoticon']:
            subset = df[df['type'] == etype]
            if len(subset) > 0:
                report += f"#### {etype.upper()} ({len(subset):,} items)\n"
                top_descriptions = subset['description'].value_counts().head(10)
                for desc, count in top_descriptions.items():
                    report += f"- {desc}: {count:,} items\n"
                report += "\n"
        
        report += """
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
"""
        
        with open('visualize/detailed_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("📄 Detailed report saved: visualize/detailed_analysis_report.md")


def main():
    """Main function"""
    
    analyzer = KaomojiEmoticonAnalyzer()
    
    # Analyze dataset
    csv_file = "../data/emoticon_descriptions_vietnamese.csv"
    results = analyzer.analyze_dataset(csv_file)
    
    print("\n" + "=" * 60)
    print("🎯 FINAL SUMMARY")
    print("=" * 60)
    
    total = results['total']
    for etype, percentage in results['percentages'].items():
        count = results['type_counts'][etype]
        print(f"{etype.upper():<12}: {count:>7,}/{total:,} ({percentage:5.1f}%)")
    
    print(f"\n📁 All results saved in: research/visualize/")


if __name__ == "__main__":
    main()
