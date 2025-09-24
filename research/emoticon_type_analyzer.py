#!/usr/bin/env python3
"""
Emoticon Type Detection and Analysis
Classify emoticons into Vietnamese, Western, and Kaomoji types
"""

import pandas as pd
import re
from typing import Tuple, Dict
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class EmoticonTypeDetector:
    def __init__(self):
        # Vietnamese emoticon patterns (multiple parentheses)
        self.vietnamese_patterns = [
            r':[\)\(DPpvV]+',  # :))), :((, :DD, :PP, :vv
            r':\d+',           # :3, :0, etc.
            r'[xX][\dD]+',     # xD, X3, etc.
        ]
        
        # Western emoticon patterns (single characters)
        self.western_patterns = [
            r':\)',   # :)
            r':\(',   # :(
            r':D',    # :D
            r':P',    # :P
            r':o',    # :o
            r';\)',   # ;)
            r':\|',   # :|
            r':/',    # :/
            r':\*',   # :*
            r'<3',    # <3
            r'>\.<',  # >.<
            r'=\)',   # =)
            r'=\(',   # =(
            r'8\)',   # 8)
            r'B\)',   # B)
        ]
        
        # Kaomoji patterns (Japanese style with special characters)
        self.kaomoji_patterns = [
            r'[（\(][^)]*[ಠಥ◕⌒´`~‾°º○●◎⊙＊✧★☆♪♫♬♡♥💕💖💗💘💙💚💛💜🤍🖤♦♧♠♤♥♡◊◈◇◆■□▪▫▲△▼▽◄►▶◀♀♂⚡⭐🌟✨💫⭐🌠☄️⚪⚫🔴🔵🔶🔷🔸🔹🔺🔻◼️◽️▪️▫️🔳🔲⬛⬜🟫🟪🟦🟩🟨🟧🟥][^)]*[）\)]',
            r'[（\(][^)]*[ωοΩ][^)]*[）\)]',  # ω character
            r'[（\(][^)]*[╯╰╭╮╱╲╳╴╵╶╷][^)]*[）\)]',  # Box drawing characters
            r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]',  # Japanese characters
            r'[ಠ_ಠಥ_ಥ◕_◕⌒_⌒´_`~_~‾_‾°_°º_º○_○●_●◎_◎⊙_⊙]',  # Face elements
            r'[\(\[][\s\S]*[ツシ][\s\S]*[\)\]]',  # Contains ツ or シ
            r'¯\\\_\(ツ\)_\/¯',  # Specific shrug kaomoji
        ]
    
    def detect_emoticon_type(self, emoticon: str) -> str:
        """Detect the type of emoticon: vietnamese, western, or kaomoji"""
        emoticon = emoticon.strip()
        
        # Check for Vietnamese patterns (multiple chars)
        for pattern in self.vietnamese_patterns:
            if re.search(pattern, emoticon):
                return 'vietnamese'
        
        # Check for Kaomoji patterns (Japanese style)
        for pattern in self.kaomoji_patterns:
            if re.search(pattern, emoticon, re.UNICODE):
                return 'kaomoji'
        
        # Check for Western patterns (simple ASCII)
        for pattern in self.western_patterns:
            if re.search(pattern, emoticon):
                return 'western'
        
        # Additional heuristics
        
        # If contains Japanese/Unicode characters, likely kaomoji
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', emoticon):
            return 'kaomoji'
        
        # If has complex structure with special chars, likely kaomoji  
        special_chars = len(re.findall(r'[^\w\s\(\):\;\-=<>]', emoticon))
        if special_chars >= 3:
            return 'kaomoji'
        
        # If has parentheses with complex content, likely kaomoji
        if re.search(r'[（\(][^)]{3,}[）\)]', emoticon):
            return 'kaomoji'
        
        # If multiple repeated chars (Vietnamese style)
        if re.search(r'[:;=][)(\\/DPpvV]{2,}', emoticon):
            return 'vietnamese'
        
        # If simple ASCII pattern, western
        if re.search(r'^[:;=][)(\\/DPpvV]$', emoticon):
            return 'western'
        
        # Default to kaomoji for complex patterns
        if len(emoticon) > 5:
            return 'kaomoji'
        
        return 'western'  # Default fallback
    
    def analyze_emoticon_dataset(self, csv_file: str) -> Dict:
        """Analyze emoticon types in the dataset"""
        
        print("🔍 Analyzing Emoticon Types in Dataset")
        print("=" * 50)
        
        # Load dataset
        df = pd.read_csv(csv_file)
        print(f"📊 Total emoticons: {len(df)}")
        
        # Detect types for each emoticon
        df['emoticon_type'] = df['emoticon_code'].apply(self.detect_emoticon_type)
        
        # Count by type
        type_counts = df['emoticon_type'].value_counts()
        total = len(df)
        
        print("\n📈 Emoticon Type Distribution:")
        for emoticon_type, count in type_counts.items():
            percentage = (count / total) * 100
            print(f"   {emoticon_type.upper()}: {count:,} ({percentage:.1f}%)")
        
        # Show examples for each type
        print("\n📝 Examples by Type:")
        for emoticon_type in ['vietnamese', 'western', 'kaomoji']:
            if emoticon_type in type_counts:
                examples = df[df['emoticon_type'] == emoticon_type]['emoticon_code'].head(10).tolist()
                print(f"\n{emoticon_type.upper()} examples:")
                for example in examples:
                    print(f"   {example}")
        
        # Analyze by description
        print("\n📋 Most Common Descriptions by Type:")
        for emoticon_type in ['vietnamese', 'western', 'kaomoji']:
            if emoticon_type in type_counts:
                subset = df[df['emoticon_type'] == emoticon_type]
                top_descriptions = subset['description'].value_counts().head(5)
                print(f"\n{emoticon_type.upper()}:")
                for desc, count in top_descriptions.items():
                    print(f"   {desc}: {count} emoticons")
        
        # Save results
        output_file = csv_file.replace('.csv', '_with_types.csv')
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n💾 Saved analysis to: {output_file}")
        
        # Create visualizations
        self.create_pie_chart(type_counts, csv_file)
        
        return {
            'total': total,
            'type_counts': type_counts.to_dict(),
            'percentages': {k: (v/total)*100 for k, v in type_counts.to_dict().items()}
        }
    
    def create_pie_chart(self, type_counts, csv_file: str):
        """Create pie chart visualization for emoticon types"""
        
        # Set up the plot
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Colors for each type
        colors = {
            'kaomoji': '#FF6B6B',     # Red
            'vietnamese': '#4ECDC4',  # Teal  
            'western': '#45B7D1'      # Blue
        }
        
        # Prepare data
        labels = []
        sizes = []
        chart_colors = []
        
        for emoticon_type, count in type_counts.items():
            labels.append(f"{emoticon_type.upper()}\n({count:,})")
            sizes.append(count)
            chart_colors.append(colors.get(emoticon_type, '#95A5A6'))
        
        # Pie chart 1: With counts
        wedges1, texts1, autotexts1 = ax1.pie(
            sizes, 
            labels=labels,
            colors=chart_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        ax1.set_title('🎯 Emoticon Type Distribution\n(With Counts)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Make percentage text more visible
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        # Pie chart 2: Clean percentages only
        wedges2, texts2, autotexts2 = ax2.pie(
            sizes,
            labels=[t.upper() for t in type_counts.keys()],
            colors=chart_colors,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 14, 'fontweight': 'bold'}
        )
        
        ax2.set_title('📊 Percentage Distribution\n(Clean View)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        # Make percentage text more visible
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontsize(16)
            autotext.set_fontweight('bold')
        
        # Add total count as subtitle
        total = sum(sizes)
        fig.suptitle(f'Vietnamese Emoticon Dataset Analysis\nTotal: {total:,} emoticons', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Add legend with statistics
        legend_labels = []
        for emoticon_type, count in type_counts.items():
            percentage = (count / total) * 100
            legend_labels.append(f"{emoticon_type.upper()}: {count:,} ({percentage:.1f}%)")
        
        fig.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=3, fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = csv_file.replace('.csv', '_type_distribution.png')
        plt.savefig(chart_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Pie chart saved to: {chart_file}")
        
        # Show the plot
        plt.show()
        plt.close()


def main():
    """Main function to analyze emoticon types"""
    
    detector = EmoticonTypeDetector()
    
    # Analyze the Vietnamese emoticon dataset
    csv_file = "../data/emoticon_descriptions_vietnamese.csv"
    results = detector.analyze_emoticon_dataset(csv_file)
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    total = results['total']
    for emoticon_type, percentage in results['percentages'].items():
        count = results['type_counts'][emoticon_type]
        print(f"{emoticon_type.upper()}: {count:,}/{total:,} ({percentage:.1f}%)")
    
    # Also analyze clean dataset if exists
    try:
        clean_csv_file = "../data/emoticon_descriptions_vietnamese_clean.csv"
        print(f"\n🧹 Analyzing Clean Dataset:")
        clean_results = detector.analyze_emoticon_dataset(clean_csv_file)
    except FileNotFoundError:
        print("Clean dataset not found, skipping...")


if __name__ == "__main__":
    main()
