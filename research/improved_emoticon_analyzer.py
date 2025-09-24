#!/usr/bin/env python3
"""
Improved Emoticon Type Classification and Visualization
Classify emoticons into Vietnamese, Western, and Kaomoji types with better logic
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Tuple, Dict, List
from collections import Counter


class ImprovedEmoticonClassifier:
    def __init__(self):
        self.setup_patterns()
    
    def setup_patterns(self):
        """Setup improved classification patterns"""
        
        # Vietnamese emoticon patterns (style phổ biến ở VN)
        self.vietnamese_patterns = [
            r':[\)\(]{2,}',        # :)), :((, :))), etc.
            r':[DPpvV]{2,}',       # :DD, :PP, :vv, etc.
            r':[0-9]+',            # :3, :0, :9, etc.
            r'[xX][DdPp]+',        # xD, XD, xP, etc.
            r'=[\)\(]{2,}',        # =)), =(((, etc.
            r'T[._-]*T',           # T.T, T_T, T-T
            r'[><=][._-]*[<>=]',   # >.<, =.=, <3, etc.
        ]
        
        # Western emoticon patterns (ASCII đơn giản)
        self.western_patterns = [
            r'^:\)$',     r'^:\($',     r'^:D$',      r'^:P$',     r'^:p$',
            r'^:o$',      r'^:O$',      r'^;\)$',     r'^:\|$',    r'^:/$',
            r'^:\*$',     r'^<3$',      r'^>.<$',     r'^=\)$',    r'^=\($',
            r'^8\)$',     r'^B\)$',     r'^:-\)$',    r'^:-\($',   r'^:-D$',
            r'^:-P$',     r'^:-p$',     r'^:-o$',     r'^:-O$',    r'^;-\)$',
            r'^:-\|$',    r'^:-/$',     r'^:-\*$',    r'^B-\)$',   r'^8-\)$',
        ]
        
        # Kaomoji indicators (Japanese style với special characters)
        self.kaomoji_indicators = [
            # Japanese/Unicode characters
            r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',  # Hiragana, Katakana, Kanji
            # Special face characters
            r'[ಠಥ◕⌒´`~‾°º○●◎⊙＊✧★☆ツシω]',
            # Complex parentheses
            r'[（）]',
            # Box drawing and special symbols
            r'[╯╰╭╮╱╲╳╴╵╶╷┬─┳┷┻━┃┏┓┗┛├┤┼]',
            # Arrows and geometric shapes  
            r'[◄►▶◀▲△▼▽◊◈◇◆■□▪▫]',
            # Hearts and decorative symbols
            r'[♡♥💕💖💗💘💙💚💛💜🤍🖤♦♧♠♤]',
            # Face elements specific to kaomoji
            r'[ᵕᴗᴖᵔᴥεッㅂㅇㅅㅁㅜㅠ]',
        ]
    
    def is_vietnamese_style(self, emoticon: str) -> bool:
        """Check if emoticon follows Vietnamese style patterns"""
        emoticon = emoticon.strip()
        
        for pattern in self.vietnamese_patterns:
            if re.search(pattern, emoticon):
                return True
        return False
    
    def is_western_style(self, emoticon: str) -> bool:
        """Check if emoticon is simple Western ASCII style"""
        emoticon = emoticon.strip()
        
        # Must be simple (short length)
        if len(emoticon) > 4:
            return False
        
        for pattern in self.western_patterns:
            if re.search(pattern, emoticon):
                return True
        
        # Additional simple western patterns
        if re.match(r'^[:;=][-)(\\/DPpOo]$', emoticon):
            return True
            
        return False
    
    def is_kaomoji_style(self, emoticon: str) -> bool:
        """Check if emoticon is Kaomoji (Japanese) style"""
        emoticon = emoticon.strip()
        
        # Count kaomoji indicators
        indicator_count = 0
        for pattern in self.kaomoji_indicators:
            if re.search(pattern, emoticon):
                indicator_count += 1
        
        # If has multiple indicators, likely kaomoji
        if indicator_count >= 2:
            return True
        
        # Single indicator but complex structure
        if indicator_count >= 1 and len(emoticon) > 3:
            return True
        
        # Complex structure with parentheses
        if re.search(r'[（\(][^)]{3,}[）\)]', emoticon):
            return True
            
        # Has non-ASCII characters
        if re.search(r'[^\x00-\x7F]', emoticon):
            return True
        
        # Long and complex
        if len(emoticon) > 6:
            return True
            
        return False
    
    def classify_emoticon(self, emoticon: str) -> str:
        """Classify emoticon into type with improved logic"""
        emoticon = emoticon.strip()
        
        # Handle edge cases
        if not emoticon:
            return 'unknown'
        
        # Check in order of specificity
        
        # 1. Vietnamese style (most specific patterns first)
        if self.is_vietnamese_style(emoticon):
            return 'vietnamese'
        
        # 2. Simple Western style
        if self.is_western_style(emoticon):
            return 'western'
        
        # 3. Kaomoji (default for complex patterns)
        if self.is_kaomoji_style(emoticon):
            return 'kaomoji'
        
        # 4. Fallback classification
        # Short and simple ASCII -> Western
        if len(emoticon) <= 3 and re.match(r'^[:\-;=)(DPpOo<>|/\\*]+$', emoticon):
            return 'western'
        
        # Everything else -> Kaomoji
        return 'kaomoji'
    
    def analyze_and_visualize(self, csv_file: str) -> Dict:
        """Analyze emoticon types and create visualizations"""
        
        print("🔍 IMPROVED EMOTICON TYPE ANALYSIS")
        print("=" * 60)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        print(f"📊 Total emoticons: {len(df):,}")
        
        # Classify each emoticon
        print("🔄 Classifying emoticons...")
        df['emoticon_type'] = df['emoticon_code'].apply(self.classify_emoticon)
        
        # Count by type
        type_counts = df['emoticon_type'].value_counts()
        total = len(df)
        
        print("\n📈 EMOTICON TYPE DISTRIBUTION:")
        for etype, count in type_counts.items():
            percentage = (count / total) * 100
            print(f"   {etype.upper():<12}: {count:>7,} emoticons ({percentage:5.1f}%)")
        
        # Create visualizations
        self.create_pie_chart(type_counts, total)
        self.create_bar_chart(type_counts)
        
        # Show examples
        self.show_examples(df)
        
        # Save results
        output_file = csv_file.replace('.csv', '_classified.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Saved classified data to: {output_file}")
        
        return {
            'total': total,
            'type_counts': type_counts.to_dict(),
            'percentages': {k: (v/total)*100 for k, v in type_counts.to_dict().items()},
            'dataframe': df
        }
    
    def create_pie_chart(self, type_counts: pd.Series, total: int):
        """Create pie chart for emoticon types"""
        
        # Prepare data
        types = type_counts.index.tolist()
        counts = type_counts.values.tolist()
        percentages = [(count/total)*100 for count in counts]
        
        # Colors for different types
        colors = {
            'kaomoji': '#FF6B6B',      # Red
            'western': '#4ECDC4',      # Teal  
            'vietnamese': '#45B7D1',   # Blue
            'unknown': '#96CEB4'       # Green
        }
        
        chart_colors = [colors.get(t.lower(), '#CCCCCC') for t in types]
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=[f'{t.upper()}\n{p:.1f}%' for t, p in zip(types, percentages)],
            colors=chart_colors,
            autopct='%1.0f',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
        )
        
        # Customize
        ax.set_title('🎯 Vietnamese Emoticon Dataset - Type Distribution\n📊 Total: {:,} Emoticons'.format(total), 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend with counts
        legend_labels = [f'{t.upper()}: {c:,} emoticons' for t, c in zip(types, counts)]
        ax.legend(wedges, legend_labels, title="Emoticon Types", 
                 loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Save pie chart
        output_path = '../visualizations/emoticon_types_pie_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Pie chart saved to: {output_path}")
        
        plt.show()
        plt.close()
    
    def create_bar_chart(self, type_counts: pd.Series):
        """Create bar chart for emoticon types"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = ax.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
        
        # Add value labels on bars
        for bar, count in zip(bars, type_counts.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('📊 Emoticon Types Distribution (Bar Chart)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Emoticon Type', fontweight='bold')
        ax.set_ylabel('Number of Emoticons', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save bar chart
        output_path = '../visualizations/emoticon_types_bar_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Bar chart saved to: {output_path}")
        
        plt.show()
        plt.close()
    
    def show_examples(self, df: pd.DataFrame):
        """Show examples of each emoticon type"""
        
        print("\n📝 EXAMPLES BY TYPE:")
        
        for etype in ['vietnamese', 'western', 'kaomoji']:
            subset = df[df['emoticon_type'] == etype]
            if len(subset) > 0:
                examples = subset['emoticon_code'].head(15).tolist()
                print(f"\n{etype.upper()} ({len(subset):,} total):")
                for i, example in enumerate(examples, 1):
                    print(f"   {i:2}. {example}")
        
        # Show statistics by description
        print("\n📋 TOP DESCRIPTIONS BY TYPE:")
        
        for etype in ['vietnamese', 'western', 'kaomoji']:
            subset = df[df['emoticon_type'] == etype]
            if len(subset) > 0:
                top_descriptions = subset['description'].value_counts().head(5)
                print(f"\n{etype.upper()}:")
                for desc, count in top_descriptions.items():
                    print(f"   {desc}: {count:,} emoticons")


def main():
    """Main function to run improved analysis"""
    
    classifier = ImprovedEmoticonClassifier()
    
    # Analyze the Vietnamese emoticon dataset
    csv_file = "../data/emoticon_descriptions_vietnamese.csv"
    results = classifier.analyze_and_visualize(csv_file)
    
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    
    total = results['total']
    for etype, percentage in results['percentages'].items():
        count = results['type_counts'][etype]
        print(f"{etype.upper():<12}: {count:>7,}/{total:,} ({percentage:5.1f}%)")


if __name__ == "__main__":
    main()
