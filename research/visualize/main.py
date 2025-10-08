import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Modern UX/UI styling
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'


def detect_emotion_type(text):
    """Detect if text contains emoticon, emoji, or no emotion."""
    # Emoticon patterns (kaomoji and western emoticons)
    emoticon_pattern = r'[;:=xX][\-o\*\']?[\)\]\(\[dDpPoO/\\|}{@><3]|[\(\)\[\]dDpPoO/\\|}{@><3][\-o\*\']?[;:=xX]|\([\^\_\-o\.][\_\-o\.]*\)|\^[\^\_\-]*\^|>[\.<]|<[\.<]|[oO][\._][oO]|[TT][\._][TT]|¯\\\_\(ツ\)\_\/¯|ಠ\_ಠ|[ಥ][\_][ಥ]|\(╯°□°\)╯︵ ┻━┻|ʕ•ᴥ•ʔ'
    
    # Emoji pattern (Unicode ranges for emojis)
    emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    
    has_emoticon = bool(re.search(emoticon_pattern, text))
    has_emoji = bool(re.search(emoji_pattern, text))
    
    if has_emoticon:
        return 'emoticon'
    elif has_emoji:
        return 'emoji'
    else:
        return 'no_emotion'


def draw_label_distribution(dataset, title="Label Distribution", save_path="./output/label_distribution.png"):
    """Draw sentiment label distribution using modern pie chart."""
    import os
    
    # Extract labels
    labels = [item['sentiment'] for item in dataset['train']]
    label_counts = Counter(labels)
    
    # Prepare data
    label_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']  # Red, Gray, Green
    
    sorted_labels = sorted(label_counts.keys())
    names = [label_map[label] for label in sorted_labels]
    counts = [label_counts[label] for label in sorted_labels]
    total = sum(counts)
    
    # Create figure with donut chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart with modern design
    wedges, texts, autotexts = ax1.pie(counts, labels=names, colors=colors, autopct='%1.1f%%',
                                        startangle=90, pctdistance=0.85,
                                        textprops={'fontsize': 13, 'weight': 'bold', 'color': '#2c3e50'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 3, 'antialiased': True})
    
    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax1.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', 
                  color='#2c3e50', pad=20)
    
    # Detailed bar chart for comparison
    bars = ax2.barh(names, counts, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        pct = (count/total)*100
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'  {count:,} ({pct:.1f}%)',
                ha='left', va='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    ax2.set_title('Count Comparison', fontsize=16, fontweight='bold', 
                  color='#2c3e50', pad=20)
    ax2.set_xlabel('Number of Samples', fontsize=13, fontweight='bold', color='#34495e')
    ax2.xaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add total count
    fig.text(0.5, 0.02, f'Total Samples: {total:,}', ha='center', va='bottom', 
             fontsize=12, color='#7f8c8d', style='italic', weight='bold')
    
    plt.suptitle(title, fontsize=18, fontweight='bold', color='#2c3e50', y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")



def draw_emotion_distribution(dataset, title="Emotion Distribution", save_path="./output/emotion_distribution.png"):
    """Draw emotion type distribution (emoticon/emoji/no_emotion) with sentiment breakdown."""
    import os
    import pandas as pd
    
    # Classify each text
    data = []
    for item in dataset['train']:
        emotion_type = detect_emotion_type(item['sentence'])
        sentiment = item['sentiment']
        data.append({
            'emotion_type': emotion_type,
            'sentiment': sentiment,
            'text': item['sentence']
        })
    
    df = pd.DataFrame(data)
    
    # Count emotion types
    emotion_counts = Counter(df['emotion_type'])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    # Color schemes
    emotion_colors = {'emoticon': '#9b59b6', 'emoji': '#f39c12', 'no_emotion': '#95a5a6'}
    sentiment_colors = {-1: '#e74c3c', 0: '#95a5a6', 1: '#2ecc71'}
    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    # 1. Main pie chart - Emotion type distribution
    ax1 = fig.add_subplot(gs[0, :2])
    
    emotion_names = ['Emoticon', 'Emoji', 'No Emotion']
    emotion_keys = ['emoticon', 'emoji', 'no_emotion']
    counts = [emotion_counts[key] for key in emotion_keys]
    colors_list = [emotion_colors[key] for key in emotion_keys]
    total = sum(counts)
    
    wedges, texts, autotexts = ax1.pie(counts, labels=emotion_names, colors=colors_list,
                                        autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                        textprops={'fontsize': 13, 'weight': 'bold', 'color': '#2c3e50'},
                                        wedgeprops={'edgecolor': 'white', 'linewidth': 3})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax1.set_title('Emotion Type Distribution', fontsize=16, fontweight='bold', 
                  color='#2c3e50', pad=20)
    
    # Add count annotations
    for i, (name, count) in enumerate(zip(emotion_names, counts)):
        pct = (count/total)*100
        ax1.text(0.95, 0.95-i*0.08, f'{name}: {count:,} ({pct:.1f}%)',
                transform=ax1.transAxes, fontsize=11, color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors_list[i], alpha=0.3))
    
    # 2. Stacked bar - Sentiment breakdown by emotion type
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Prepare stacked data
    emotion_sentiment_data = {}
    for emo_type in emotion_keys:
        emotion_sentiment_data[emo_type] = {-1: 0, 0: 0, 1: 0}
    
    for _, row in df.iterrows():
        emotion_sentiment_data[row['emotion_type']][row['sentiment']] += 1
    
    x_labels = ['Emoticon', 'Emoji', 'No Emotion']
    neg_counts = [emotion_sentiment_data[key][-1] for key in emotion_keys]
    neu_counts = [emotion_sentiment_data[key][0] for key in emotion_keys]
    pos_counts = [emotion_sentiment_data[key][1] for key in emotion_keys]
    
    x_pos = range(len(x_labels))
    ax2.bar(x_pos, neg_counts, color=sentiment_colors[-1], label='Negative', alpha=0.85)
    ax2.bar(x_pos, neu_counts, bottom=neg_counts, color=sentiment_colors[0], 
            label='Neutral', alpha=0.85)
    ax2.bar(x_pos, pos_counts, bottom=[n+ne for n, ne in zip(neg_counts, neu_counts)],
            color=sentiment_colors[1], label='Positive', alpha=0.85)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=15, ha='right')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold', color='#34495e')
    ax2.set_title('Sentiment by Emotion Type', fontsize=14, fontweight='bold', 
                  color='#2c3e50', pad=15)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3-5. Individual pie charts for each emotion type's sentiment distribution
    for idx, (emo_key, emo_name) in enumerate(zip(emotion_keys, emotion_names)):
        ax = fig.add_subplot(gs[1, idx])
        
        sent_counts = [emotion_sentiment_data[emo_key][s] for s in [-1, 0, 1]]
        sent_names = ['Neg', 'Neu', 'Pos']
        sent_colors = [sentiment_colors[s] for s in [-1, 0, 1]]
        
        # Only show non-zero slices
        non_zero_idx = [i for i, c in enumerate(sent_counts) if c > 0]
        if non_zero_idx:
            filtered_counts = [sent_counts[i] for i in non_zero_idx]
            filtered_names = [sent_names[i] for i in non_zero_idx]
            filtered_colors = [sent_colors[i] for i in non_zero_idx]
            
            wedges, texts, autotexts = ax.pie(filtered_counts, labels=filtered_names,
                                              colors=filtered_colors, autopct='%1.1f%%',
                                              startangle=90, textprops={'fontsize': 10, 'weight': 'bold'},
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(11)
        
        ax.set_title(f'{emo_name}\nSentiment', fontsize=12, fontweight='bold', 
                    color=emotion_colors[emo_key], pad=10)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', color='#2c3e50', y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")



def draw_length_distribution(dataset, title="Length Distribution", save_path="./output/length_distribution.png"):
    """Draw comprehensive text length analysis with creative insights."""
    import os
    import pandas as pd
    import numpy as np
    
    # Prepare data with emotion detection
    data = []
    for item in dataset['train']:
        text = item['sentence']
        data.append({
            'length': len(text),
            'word_count': len(text.split()),
            'sentiment': item['sentiment'],
            'emotion_type': detect_emotion_type(text)
        })
    df = pd.DataFrame(data)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
    
    # Color schemes
    sentiment_colors = {-1: '#e74c3c', 0: '#95a5a6', 1: '#2ecc71'}
    sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    emotion_colors = {'emoticon': '#9b59b6', 'emoji': '#f39c12', 'no_emotion': '#95a5a6'}
    
    # 1. Overall character length distribution with KDE (top row, spanning 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    sns.histplot(data=df, x='length', bins=60, kde=True, color='#3498db', 
                 alpha=0.6, edgecolor='white', linewidth=1, ax=ax1)
    
    mean_len = df['length'].mean()
    median_len = df['length'].median()
    ax1.axvline(mean_len, color='#e74c3c', linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_len:.0f}')
    ax1.axvline(median_len, color='#2ecc71', linestyle='--', linewidth=2.5, 
                label=f'Median: {median_len:.0f}')
    
    ax1.set_title('Character Length Distribution', fontsize=15, fontweight='bold', 
                  color='#2c3e50', pad=15)
    ax1.set_xlabel('Characters', fontsize=12, fontweight='bold', color='#34495e')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='#34495e')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. Word count distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    sns.histplot(data=df, x='word_count', bins=40, kde=True, color='#e67e22', 
                 alpha=0.6, edgecolor='white', linewidth=1, ax=ax2)
    
    mean_words = df['word_count'].mean()
    ax2.axvline(mean_words, color='#c0392b', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_words:.1f}')
    
    ax2.set_title('Word Count Distribution', fontsize=15, fontweight='bold', 
                  color='#2c3e50', pad=15)
    ax2.set_xlabel('Words', fontsize=12, fontweight='bold', color='#34495e')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='#34495e')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. Length by sentiment - Violin plot (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    sent_order = sorted(df['sentiment'].unique())
    palette = [sentiment_colors[s] for s in sent_order]
    
    parts = ax3.violinplot([df[df['sentiment'] == s]['length'].values for s in sent_order],
                           positions=sent_order, widths=0.7, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')
        pc.set_linewidth(2)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            parts[partname].set_edgecolor('#2c3e50')
            parts[partname].set_linewidth(1.5)
    
    ax3.set_xticks(sent_order)
    ax3.set_xticklabels([sentiment_labels[s] for s in sent_order], fontsize=10)
    ax3.set_title('Length by Sentiment', fontsize=13, fontweight='bold', 
                  color='#2c3e50', pad=12)
    ax3.set_ylabel('Characters', fontsize=11, fontweight='bold', color='#34495e')
    ax3.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax3.set_axisbelow(True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 4. Length by emotion type - Box plot (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    emotion_order = ['emoticon', 'emoji', 'no_emotion']
    emotion_names = ['Emoticon', 'Emoji', 'No Emotion']
    
    bp = ax4.boxplot([df[df['emotion_type'] == e]['length'].values for e in emotion_order],
                     tick_labels=emotion_names, patch_artist=True, widths=0.6)
    
    for patch, emo in zip(bp['boxes'], emotion_order):
        patch.set_facecolor(emotion_colors[emo])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
        patch.set_linewidth(2)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='#2c3e50', linewidth=1.5)
    
    ax4.set_title('Length by Emotion Type', fontsize=13, fontweight='bold', 
                  color='#2c3e50', pad=12)
    ax4.set_ylabel('Characters', fontsize=11, fontweight='bold', color='#34495e')
    ax4.set_xticklabels(emotion_names, rotation=15, ha='right', fontsize=9)
    ax4.yaxis.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax4.set_axisbelow(True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 5. Scatter: Length vs Word Count colored by sentiment (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    for sent in sorted(df['sentiment'].unique()):
        sent_df = df[df['sentiment'] == sent]
        ax5.scatter(sent_df['word_count'], sent_df['length'], 
                   c=sentiment_colors[sent], label=sentiment_labels[sent],
                   alpha=0.5, s=20, edgecolors='white', linewidth=0.5)
    
    ax5.set_title('Length vs Word Count', fontsize=13, fontweight='bold', 
                  color='#2c3e50', pad=12)
    ax5.set_xlabel('Word Count', fontsize=11, fontweight='bold', color='#34495e')
    ax5.set_ylabel('Characters', fontsize=11, fontweight='bold', color='#34495e')
    ax5.legend(loc='upper left', fontsize=9, framealpha=0.9, markerscale=1.5)
    ax5.grid(True, linestyle='--', alpha=0.3, color='#7f8c8d')
    ax5.set_axisbelow(True)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    
    # 6. Statistics table (bottom, spanning all columns)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Calculate comprehensive statistics
    stats_rows = []
    
    # Overall stats
    stats_rows.append(['Overall', f"{df['length'].mean():.0f}", f"{df['length'].median():.0f}",
                      f"{df['length'].std():.0f}", f"{df['word_count'].mean():.1f}",
                      f"{df['length'].min():.0f}", f"{df['length'].max():.0f}"])
    
    # By sentiment
    for sent in sorted(df['sentiment'].unique()):
        sent_df = df[df['sentiment'] == sent]
        stats_rows.append([
            sentiment_labels[sent],
            f"{sent_df['length'].mean():.0f}",
            f"{sent_df['length'].median():.0f}",
            f"{sent_df['length'].std():.0f}",
            f"{sent_df['word_count'].mean():.1f}",
            f"{sent_df['length'].min():.0f}",
            f"{sent_df['length'].max():.0f}"
        ])
    
    # By emotion type
    for emo in emotion_order:
        emo_df = df[df['emotion_type'] == emo]
        emo_name = {'emoticon': 'Emoticon', 'emoji': 'Emoji', 'no_emotion': 'No Emotion'}[emo]
        stats_rows.append([
            emo_name,
            f"{emo_df['length'].mean():.0f}",
            f"{emo_df['length'].median():.0f}",
            f"{emo_df['length'].std():.0f}",
            f"{emo_df['word_count'].mean():.1f}",
            f"{emo_df['length'].min():.0f}",
            f"{emo_df['length'].max():.0f}"
        ])
    
    table = ax6.table(cellText=stats_rows,
                     colLabels=['Category', 'Mean Len', 'Median Len', 'Std', 'Mean Words', 'Min', 'Max'],
                     cellLoc='center', loc='center',
                     colWidths=[0.18, 0.14, 0.14, 0.12, 0.14, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style table
    for i in range(len(stats_rows) + 1):
        for j in range(7):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495e')
                cell.set_text_props(weight='bold', color='white')
            else:
                # Color based on category
                category = stats_rows[i-1][0]
                if category in sentiment_labels.values():
                    sent_key = [k for k, v in sentiment_labels.items() if v == category][0]
                    cell.set_facecolor(sentiment_colors[sent_key])
                    cell.set_alpha(0.15)
                elif category in ['Emoticon', 'Emoji', 'No Emotion']:
                    emo_key = {'Emoticon': 'emoticon', 'Emoji': 'emoji', 'No Emotion': 'no_emotion'}[category]
                    cell.set_facecolor(emotion_colors[emo_key])
                    cell.set_alpha(0.15)
                else:
                    cell.set_facecolor('#3498db')
                    cell.set_alpha(0.1)
                
                cell.set_text_props(color='#2c3e50')
                if j == 0:
                    cell.set_text_props(weight='bold')
    
    ax6.set_title('Comprehensive Statistics Summary', fontsize=14, fontweight='bold', 
                  color='#2c3e50', pad=15)
    
    plt.suptitle(title, fontsize=18, fontweight='bold', color='#2c3e50', y=0.98)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")



from utils import load_dataset, load_config

if __name__ == '__main__':
    config = load_config("../config.yaml")
    dataset = load_dataset(config.datasets.social_comments.path)
    
    draw_label_distribution(dataset, title="Original Dataset Label Distribution", save_path="./visualize/output/original_label_distribution.png")
    draw_emotion_distribution(dataset, title="Original Dataset Emotion Distribution", save_path="./visualize/output/original_emotion_distribution.png")
    draw_length_distribution(dataset, title="Original Dataset Length Distribution", save_path="./visualize/output/original_length_distribution.png")
