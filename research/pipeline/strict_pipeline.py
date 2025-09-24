"""
Strict Vietnamese Emoticon/Kaomoji Data Pipeline
Only text-based emoticons and kaomoji, NO emoji images
"""

import pandas as pd
import os
from typing import List, Dict, Tuple
import re
from collections import Counter
from datasets import load_dataset as ds_load_dataset
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
from strict_emoticon_detector import StrictEmoticonDetector

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrictVietnameseEmoticonPipeline:
    def __init__(self):
        """Initialize the strict pipeline"""
        load_dotenv("../.env")
        self.hf_token = os.getenv('HF_TOKEN')
        
        # Initialize strict detector
        self.detector = StrictEmoticonDetector()
        
        # Target datasets to process - EXPANDED with verified datasets
        self.dataset_configs = {
            # Original verified datasets
            'vanhai123/vietnamese-social-comments': {
                'content_cols': ['comment'],
                'sentiment_col': 'label',
                'category_col': 'category'
            },
            'minhtoan/vietnamese-comment-sentiment': {
                'content_cols': ['Title', 'Content', 'BriefContent'],
                'sentiment_col': 'Sentiment',
            },
            'HelloWorld2307/aivivn_test': {
                'content_cols': ['discriptions'], 
                'sentiment_col': 'mapped_rating'
            },
            'sepidmnorozy/Vietnamese_sentiment': {
                'content_cols': ['text'],
                'sentiment_col': 'label'
            },
            'anotherpolarbear/vietnamese-sentiment-analysis': {
                'content_cols': ['comment'],
                'sentiment_col': 'label'
            },
            'iaiuet/banking_sentiment_vietnamese': {
                'content_cols': ['text_paragraph'],
                'sentiment_col': 'label'
            },
            
            # Additional verified datasets from HuggingFace search
            'another-symato/VMTEB-vietnamese_students_feedback_sentiment': {
                'content_cols': ['text', 'feedback', 'content'],
                'sentiment_col': 'label'
            },
            # Try some other potential formats
            'uit-nlp/vietnamese_students_feedback': {
                'content_cols': ['text', 'feedback'],
                'sentiment_col': 'sentiment'
            },
            'tarudesu/ViSoBERT': {
                'content_cols': ['text'],
                'sentiment_col': 'label'
            },
            'ZeroWw/Vietnamese-sentiment': {
                'content_cols': ['text'],
                'sentiment_col': 'sentiment'
            },
            # Vietnamese social media datasets
            'vitk/vietnamese-social-media-sentiment': {
                'content_cols': ['post', 'content', 'text'],
                'sentiment_col': 'sentiment'
            },
            'vietnamese-nlp/sentiment-analysis': {
                'content_cols': ['text'],
                'sentiment_col': 'label'
            }
        }
        
        self.raw_data = []
        self.filtered_data = []
        self.final_data = []
        self.analysis_results = {}
        
    def step1_collect_raw_data(self) -> None:
        """Step 1: Collect raw data from all datasets"""
        logger.info("🔄 Step 1: Collecting raw data from datasets...")
        
        self.raw_data = []
        
        for dataset_name, config in self.dataset_configs.items():
            logger.info(f"Processing {dataset_name}...")
            
            try:
                dataset = ds_load_dataset(dataset_name, token=self.hf_token)
                
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    df = split_data.to_pandas()
                    
                    logger.info(f"  {split_name}: {len(df)} samples")
                    
                    # Add metadata
                    df['dataset_source'] = dataset_name
                    df['split'] = split_name
                    
                    self.raw_data.append(df)
                    
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                logger.info(f"  Skipping {dataset_name} and continuing with next dataset...")
                continue
        
        # Combine all raw data
        if self.raw_data:
            combined_raw = pd.concat(self.raw_data, ignore_index=True)
            logger.info(f"✅ Step 1 Complete: {len(combined_raw)} total raw samples collected")
            self.raw_data = combined_raw
        else:
            logger.error("❌ No raw data collected!")
    
    def step2_strict_emoticon_filtering(self) -> None:
        """Step 2: STRICT filtering for text emoticon/kaomoji content only"""
        logger.info("🔄 Step 2: STRICT filtering for text emoticons/kaomoji only...")
        logger.info("⚠️  Excluding emoji images, false positives, and plain text")
        
        if isinstance(self.raw_data, list):
            logger.error("❌ No raw data to filter!")
            return
        
        filtered_rows = []
        emoji_rejected = 0
        false_positive_rejected = 0
        no_emoticon_rejected = 0
        
        for idx, row in self.raw_data.iterrows():
            dataset_name = row['dataset_source']
            config = self.dataset_configs[dataset_name]
            
            # Check all potential content columns
            found_valid_emoticon = False
            emoticon_content = None
            emoticon_column = None
            emoticon_type = None
            
            for col_name in config['content_cols']:
                if col_name in row and isinstance(row[col_name], str) and len(str(row[col_name]).strip()) > 0:
                    col_value = str(row[col_name]).strip()
                    
                    # Strict validation
                    is_valid, detected_type = self.detector.is_valid_emoticon_content(col_value)
                    
                    if is_valid:
                        found_valid_emoticon = True
                        emoticon_content = col_value
                        emoticon_column = col_name
                        emoticon_type = detected_type
                        break
                    elif detected_type == "emoji_image":
                        emoji_rejected += 1
                    elif detected_type == "false_positive":
                        false_positive_rejected += 1
                    elif detected_type == "no_emoticon":
                        no_emoticon_rejected += 1
            
            if found_valid_emoticon:
                # Extract sentiment
                sentiment = None
                if config['sentiment_col'] and config['sentiment_col'] in row:
                    sentiment = row[config['sentiment_col']]
                
                filtered_rows.append({
                    'content': emoticon_content,
                    'sentiment_label': self.normalize_sentiment_label(sentiment),
                    'original_sentiment': sentiment,
                    'emoticon_type': emoticon_type,
                    'source_column': emoticon_column,
                    'dataset_source': dataset_name,
                    'split': row['split']
                })
            
            # Progress logging
            if len(filtered_rows) % 500 == 0 and len(filtered_rows) > 0:
                logger.info(f"  Found {len(filtered_rows)} valid emoticon samples so far...")
        
        self.filtered_data = pd.DataFrame(filtered_rows)
        
        logger.info(f"✅ Step 2 Complete: {len(self.filtered_data)} STRICT emoticon/kaomoji samples")
        logger.info(f"   📊 Rejected: {emoji_rejected} emoji images, {false_positive_rejected} false positives, {no_emoticon_rejected} no emoticons")
    
    def normalize_sentiment_label(self, label) -> str:
        """Normalize sentiment labels"""
        if label is None or pd.isna(label):
            return None
        
        label_str = str(label).strip().lower()
        
        # Vietnamese labels
        vietnamese_mapping = {
            'tích cực': 'positive',
            'tiêu cực': 'negative',
            'trung lập': 'neutral',
            'trung tính': 'neutral'
        }
        
        if label_str in vietnamese_mapping:
            return vietnamese_mapping[label_str]
        
        # English labels
        english_mapping = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'toxic': 'negative'
        }
        
        if label_str in english_mapping:
            return english_mapping[label_str]
        
        # Numeric labels (common patterns)
        if label_str.isdigit():
            num_label = int(label_str)
            if num_label <= 1:
                return 'negative'
            elif num_label <= 3:
                return 'neutral'
            else:
                return 'positive'
        
        # Rating scale (1-5)
        try:
            float_label = float(label_str)
            if float_label <= 2:
                return 'negative'
            elif float_label <= 3:
                return 'neutral'
            else:
                return 'positive'
        except:
            pass
        
        return None
    
    def step3_remove_duplicates(self) -> None:
        """Step 3: Remove duplicate content"""
        logger.info("🔄 Step 3: Removing duplicate content...")
        
        initial_count = len(self.filtered_data)
        
        # Remove exact duplicates based on content
        self.final_data = self.filtered_data.drop_duplicates(
            subset=['content'], 
            keep='first'
        ).reset_index(drop=True)
        
        duplicates_removed = initial_count - len(self.final_data)
        logger.info(f"✅ Step 3 Complete: Removed {duplicates_removed} duplicates, {len(self.final_data)} unique samples remain")
    
    def step4_analyze_data(self) -> None:
        """Step 4: Comprehensive data analysis with visualizations"""
        logger.info("🔄 Step 4: Analyzing final dataset...")
        
        if len(self.final_data) == 0:
            logger.error("❌ No data to analyze!")
            return
        
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(self.final_data)
        analysis['timestamp'] = datetime.now().isoformat()
        
        # Sentiment distribution
        labeled_data = self.final_data[self.final_data['sentiment_label'].notna()]
        analysis['labeled_samples'] = len(labeled_data)
        analysis['unlabeled_samples'] = len(self.final_data) - len(labeled_data)
        
        if len(labeled_data) > 0:
            sentiment_dist = labeled_data['sentiment_label'].value_counts()
            analysis['sentiment_distribution'] = sentiment_dist.to_dict()
        else:
            analysis['sentiment_distribution'] = {}
        
        # Dataset source distribution
        source_dist = self.final_data['dataset_source'].value_counts()
        analysis['source_distribution'] = source_dist.to_dict()
        
        # Emoticon type distribution
        emoticon_type_dist = self.final_data['emoticon_type'].value_counts()
        analysis['emoticon_type_distribution'] = emoticon_type_dist.to_dict()
        
        # Content length analysis
        content_lengths = self.final_data['content'].str.len()
        analysis['content_length'] = {
            'mean': float(content_lengths.mean()),
            'median': float(content_lengths.median()),
            'min': int(content_lengths.min()),
            'max': int(content_lengths.max()),
            'std': float(content_lengths.std())
        }
        
        # Language characteristics
        analysis['language_stats'] = self.analyze_language_characteristics()
        
        # Emoticon pattern analysis
        analysis['emoticon_patterns'] = self.analyze_emoticon_patterns()
        
        self.analysis_results = analysis
        
        # Print summary
        self.print_analysis_summary()
        
        # Create visualizations
        self.create_visualizations()
        
        logger.info("✅ Step 4 Complete: Data analysis finished")
    
    def analyze_language_characteristics(self) -> Dict:
        """Analyze Vietnamese language characteristics"""
        total_chars = 0
        total_words = 0
        vietnamese_chars = 0
        
        vietnamese_pattern = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]')
        
        for _, row in self.final_data.iterrows():
            content = str(row['content'])
            total_chars += len(content)
            total_words += len(content.split())
            vietnamese_chars += len(vietnamese_pattern.findall(content))
        
        avg_chars_per_sample = total_chars / len(self.final_data) if len(self.final_data) > 0 else 0
        avg_words_per_sample = total_words / len(self.final_data) if len(self.final_data) > 0 else 0
        vietnamese_ratio = vietnamese_chars / total_chars if total_chars > 0 else 0
        
        return {
            'avg_chars_per_sample': round(avg_chars_per_sample, 2),
            'avg_words_per_sample': round(avg_words_per_sample, 2),
            'vietnamese_char_ratio': round(vietnamese_ratio, 3),
            'total_vietnamese_chars': vietnamese_chars
        }
    
    def analyze_emoticon_patterns(self) -> Dict:
        """Analyze specific emoticon patterns found"""
        pattern_counts = {}
        
        for _, row in self.final_data.iterrows():
            content = str(row['content'])
            emoticons = self.detector.extract_emoticons_from_text(content)
            
            for emoticon in emoticons:
                if emoticon in pattern_counts:
                    pattern_counts[emoticon] += 1
                else:
                    pattern_counts[emoticon] = 1
        
        # Get top 20 most common emoticon patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_unique_patterns': len(pattern_counts),
            'top_patterns': dict(sorted_patterns[:20]),
            'total_pattern_occurrences': sum(pattern_counts.values())
        }
    
    def create_visualizations(self) -> None:
        """Create comprehensive data visualizations and save to files"""
        logger.info("📊 Creating data visualizations...")
        
        # Create visualizations directory
        viz_dir = "../visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Sentiment Distribution Pie Chart
        self.create_sentiment_pie_chart(viz_dir)
        
        # 2. Dataset Source Distribution Bar Chart
        self.create_source_distribution_chart(viz_dir)
        
        # 3. Content Length Distribution
        self.create_content_length_distribution(viz_dir)
        
        # 4. Emoticon Pattern Analysis
        self.create_emoticon_pattern_charts(viz_dir)
        
        # 5. Emoticon Word Cloud
        self.create_emoticon_wordcloud(viz_dir)
        
        # 6. Interactive Dashboard
        self.create_interactive_dashboard(viz_dir)
        
        logger.info(f"✅ Visualizations saved to {viz_dir}/")
    
    def create_sentiment_pie_chart(self, viz_dir: str) -> None:
        """Create sentiment distribution pie chart"""
        if not self.analysis_results['sentiment_distribution']:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        sentiment_data = self.analysis_results['sentiment_distribution']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        wedges, texts, autotexts = ax1.pie(
            sentiment_data.values(), 
            labels=sentiment_data.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax1.set_title('📊 Sentiment Distribution', fontsize=16, fontweight='bold')
        
        # Bar chart
        ax2.bar(sentiment_data.keys(), sentiment_data.values(), color=colors)
        ax2.set_title('📈 Sentiment Counts', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        for i, v in enumerate(sentiment_data.values()):
            ax2.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/sentiment_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_source_distribution_chart(self, viz_dir: str) -> None:
        """Create dataset source distribution chart"""
        source_data = self.analysis_results['source_distribution']
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sources = list(source_data.keys())
        counts = list(source_data.values())
        
        # Shorten source names for better display
        short_sources = [src.split('/')[-1] for src in sources]
        
        bars = ax.barh(range(len(sources)), counts, color=sns.color_palette("viridis", len(sources)))
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(short_sources)
        ax.set_xlabel('Number of Samples')
        ax.set_title('📚 Dataset Source Distribution', fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate(counts):
            ax.text(v + max(counts)*0.01, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/source_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_content_length_distribution(self, viz_dir: str) -> None:
        """Create content length distribution charts"""
        content_lengths = self.final_data['content'].str.len()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram
        ax1.hist(content_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('📏 Content Length Distribution', fontweight='bold')
        ax1.set_xlabel('Character Count')
        ax1.set_ylabel('Frequency')
        
        # 2. Box plot
        ax2.boxplot(content_lengths, vert=True)
        ax2.set_title('📊 Content Length Box Plot', fontweight='bold')
        ax2.set_ylabel('Character Count')
        
        # 3. Log-scale histogram
        ax3.hist(content_lengths, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_yscale('log')
        ax3.set_title('📏 Content Length (Log Scale)', fontweight='bold')
        ax3.set_xlabel('Character Count')
        ax3.set_ylabel('Frequency (Log)')
        
        # 4. Statistics text
        ax4.axis('off')
        stats_text = f"""
        📈 Content Length Statistics:
        
        Mean: {content_lengths.mean():.1f} characters
        Median: {content_lengths.median():.1f} characters
        Std Dev: {content_lengths.std():.1f} characters
        Min: {content_lengths.min()} characters
        Max: {content_lengths.max()} characters
        
        Quartiles:
        Q1: {content_lengths.quantile(0.25):.1f}
        Q3: {content_lengths.quantile(0.75):.1f}
        IQR: {content_lengths.quantile(0.75) - content_lengths.quantile(0.25):.1f}
        """
        ax4.text(0.1, 0.9, stats_text, fontsize=12, va='top', ha='left', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/content_length_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_emoticon_pattern_charts(self, viz_dir: str) -> None:
        """Create emoticon pattern analysis charts"""
        pattern_data = self.analysis_results['emoticon_patterns']['top_patterns']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Top 15 emoticon patterns
        top_15 = dict(list(pattern_data.items())[:15])
        patterns = list(top_15.keys())
        counts = list(top_15.values())
        
        bars = ax1.bar(range(len(patterns)), counts, color=sns.color_palette("Set2", len(patterns)))
        ax1.set_xticks(range(len(patterns)))
        ax1.set_xticklabels(patterns, rotation=45, ha='right')
        ax1.set_title('🎭 Top 15 Emoticon Patterns', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Frequency')
        
        # Add value labels
        for i, v in enumerate(counts):
            ax1.text(i, v + max(counts)*0.01, str(v), ha='center', fontweight='bold', fontsize=10)
        
        # 2. Emoticon type distribution
        emoticon_types = self.analysis_results['emoticon_type_distribution']
        colors_pie = ['#FFD93D', '#6BCF7F']
        wedges, texts, autotexts = ax2.pie(
            emoticon_types.values(),
            labels=[t.replace('_', ' ').title() for t in emoticon_types.keys()],
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax2.set_title('🎨 Emoticon Type Distribution', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{viz_dir}/emoticon_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_emoticon_wordcloud(self, viz_dir: str) -> None:
        """Create emoticon word cloud"""
        try:
            pattern_data = self.analysis_results['emoticon_patterns']['top_patterns']
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400,
                background_color='white',
                max_words=100,
                relative_scaling=0.5,
                colormap='viridis'
            ).generate_from_frequencies(pattern_data)
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('☁️ Emoticon Pattern Cloud', fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f"{viz_dir}/emoticon_wordcloud.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create word cloud: {e}")
    
    def create_interactive_dashboard(self, viz_dir: str) -> None:
        """Create interactive dashboard with Plotly"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Sentiment Distribution', 'Dataset Sources', 'Content Length', 'Top Emoticons'],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # 1. Sentiment pie chart
            sentiment_data = self.analysis_results['sentiment_distribution']
            fig.add_trace(
                go.Pie(
                    labels=list(sentiment_data.keys()),
                    values=list(sentiment_data.values()),
                    name="Sentiment"
                ),
                row=1, col=1
            )
            
            # 2. Source distribution
            source_data = self.analysis_results['source_distribution']
            short_sources = [src.split('/')[-1] for src in source_data.keys()]
            fig.add_trace(
                go.Bar(
                    x=short_sources,
                    y=list(source_data.values()),
                    name="Sources"
                ),
                row=1, col=2
            )
            
            # 3. Content length histogram
            content_lengths = self.final_data['content'].str.len()
            fig.add_trace(
                go.Histogram(
                    x=content_lengths,
                    nbinsx=30,
                    name="Content Length"
                ),
                row=2, col=1
            )
            
            # 4. Top emoticons
            pattern_data = self.analysis_results['emoticon_patterns']['top_patterns']
            top_10 = dict(list(pattern_data.items())[:10])
            fig.add_trace(
                go.Bar(
                    x=list(top_10.keys()),
                    y=list(top_10.values()),
                    name="Emoticons"
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="🎯 Vietnamese Emoticon/Kaomoji Dataset Dashboard",
                title_font_size=20,
                showlegend=False,
                height=800
            )
            
            # Save interactive HTML
            fig.write_html(f"{viz_dir}/interactive_dashboard.html")
            
            # Save static image
            pio.write_image(fig, f"{viz_dir}/interactive_dashboard.png", width=1200, height=800)
            
        except Exception as e:
            logger.warning(f"Could not create interactive dashboard: {e}")
    
    def print_analysis_summary(self) -> None:
        """Print comprehensive analysis summary"""
        print("\n" + "="*70)
        print("📊 STRICT VIETNAMESE EMOTICON/KAOMOJI DATASET ANALYSIS")
        print("🚫 NO EMOJI IMAGES - TEXT EMOTICONS ONLY")
        print("="*70)
        
        print(f"\n📈 Dataset Overview:")
        print(f"   Total samples: {self.analysis_results['total_samples']:,}")
        print(f"   Labeled samples: {self.analysis_results['labeled_samples']:,}")
        print(f"   Unlabeled samples: {self.analysis_results['unlabeled_samples']:,}")
        
        if self.analysis_results['sentiment_distribution']:
            print(f"\n🏷️ Sentiment Distribution:")
            for sentiment, count in self.analysis_results['sentiment_distribution'].items():
                percentage = (count / self.analysis_results['labeled_samples']) * 100
                print(f"   {sentiment}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n📚 Dataset Sources:")
        for source, count in self.analysis_results['source_distribution'].items():
            percentage = (count / self.analysis_results['total_samples']) * 100
            print(f"   {source}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n🎭 Emoticon Types (STRICT):")
        for emo_type, count in self.analysis_results['emoticon_type_distribution'].items():
            percentage = (count / self.analysis_results['total_samples']) * 100
            print(f"   {emo_type.replace('_', ' ').title()}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\n📝 Content Characteristics:")
        length_stats = self.analysis_results['content_length']
        print(f"   Average length: {length_stats['mean']:.1f} characters")
        print(f"   Median length: {length_stats['median']:.1f} characters")
        print(f"   Length range: {length_stats['min']} - {length_stats['max']} characters")
        
        print(f"\n🔤 Emoticon Patterns:")
        pattern_stats = self.analysis_results['emoticon_patterns']
        print(f"   Total unique patterns: {pattern_stats['total_unique_patterns']:,}")
        print(f"   Top 10 most common:")
        for pattern, count in list(pattern_stats['top_patterns'].items())[:10]:
            print(f"     '{pattern}': {count:,} times")
        
        print(f"\n🌏 Language Analysis:")
        lang_stats = self.analysis_results['language_stats']
        print(f"   Average words per sample: {lang_stats['avg_words_per_sample']}")
        print(f"   Vietnamese character ratio: {lang_stats['vietnamese_char_ratio']:.1%}")
        print(f"   Total Vietnamese characters: {lang_stats['total_vietnamese_chars']:,}")
    
    def save_results(self, output_dir: str = "../data") -> None:
        """Save final dataset and analysis results"""
        logger.info("💾 Saving strict pipeline results...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save final training dataset (labeled only)
        training_data = self.final_data[self.final_data['sentiment_label'].notna()]
        training_file = os.path.join(output_dir, "strict_vietnamese_emoticon_training_dataset.csv")
        training_data[['content', 'sentiment_label']].to_csv(training_file, index=False, encoding='utf-8')
        
        # Save complete dataset with metadata
        complete_file = os.path.join(output_dir, "strict_vietnamese_emoticon_complete_dataset.csv")
        self.final_data.to_csv(complete_file, index=False, encoding='utf-8')
        
        # Save analysis results
        analysis_file = os.path.join(output_dir, "strict_dataset_analysis_report.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        # Save pipeline summary
        summary_file = os.path.join(output_dir, "strict_pipeline_summary.md")
        self.generate_markdown_report(summary_file)
        
        logger.info(f"✅ Strict results saved:")
        logger.info(f"   Training dataset: {training_file}")
        logger.info(f"   Complete dataset: {complete_file}")
        logger.info(f"   Analysis report: {analysis_file}")
        logger.info(f"   Summary report: {summary_file}")
    
    def generate_markdown_report(self, output_file: str) -> None:
        """Generate comprehensive markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# STRICT Vietnamese Emoticon/Kaomoji Dataset Pipeline Report\n\n")
            f.write("🚫 **NO EMOJI IMAGES** - Text-based emoticons and kaomoji ONLY\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Strict Pipeline Overview\n\n")
            f.write("This dataset was created using a 4-step STRICT pipeline:\n")
            f.write("1. **Data Collection**: Gather raw data from multiple Vietnamese datasets\n")
            f.write("2. **STRICT Filtering**: Filter for TEXT emoticon/kaomoji content only (NO emoji images)\n")
            f.write("3. **Deduplication**: Remove duplicate content\n")
            f.write("4. **Analysis**: Comprehensive dataset analysis\n\n")
            
            f.write("## Quality Standards\n\n")
            f.write("- ✅ **Text emoticons only**: :), :D, :)), xD, @_@, uwu, etc.\n")
            f.write("- ✅ **Japanese kaomoji**: ಠ_ಠ, ◕_◕, ツ, etc.\n")
            f.write("- 🚫 **NO emoji images**: 😊, 😢, 💕 (completely filtered out)\n")
            f.write("- 🚫 **NO false positives**: Product names, dates, URLs excluded\n")
            f.write("- 🚫 **NO plain text**: Only content with actual emoticons\n\n")
            
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Samples**: {self.analysis_results['total_samples']:,}\n")
            f.write(f"- **Labeled Samples**: {self.analysis_results['labeled_samples']:,}\n")
            f.write(f"- **Unlabeled Samples**: {self.analysis_results['unlabeled_samples']:,}\n\n")
            
            if self.analysis_results['sentiment_distribution']:
                f.write("## Sentiment Distribution\n\n")
                for sentiment, count in self.analysis_results['sentiment_distribution'].items():
                    percentage = (count / self.analysis_results['labeled_samples']) * 100
                    f.write(f"- **{sentiment.title()}**: {count:,} samples ({percentage:.1f}%)\n")
                f.write("\n")
            
            f.write("## Emoticon Type Distribution\n\n")
            for emo_type, count in self.analysis_results['emoticon_type_distribution'].items():
                percentage = (count / self.analysis_results['total_samples']) * 100
                display_name = emo_type.replace('_', ' ').title()
                f.write(f"- **{display_name}**: {count:,} samples ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("## Top Emoticon Patterns\n\n")
            pattern_stats = self.analysis_results['emoticon_patterns']
            f.write(f"Total unique patterns: {pattern_stats['total_unique_patterns']:,}\n\n")
            for pattern, count in list(pattern_stats['top_patterns'].items())[:15]:
                f.write(f"- **`{pattern}`**: {count:,} occurrences\n")
            f.write("\n")
            
            f.write("## Usage\n\n")
            f.write("This STRICT dataset is perfect for:\n")
            f.write("- Training Vietnamese sentiment analysis models for text emoticons\n")
            f.write("- Research on text-based emoticon usage in Vietnamese social media\n")
            f.write("- Developing Vietnamese chatbots with text emoticon understanding\n")
            f.write("- Cross-cultural text emoticon analysis (NO emoji contamination)\n\n")
    
    def run_complete_pipeline(self) -> None:
        """Run the complete 4-step STRICT pipeline"""
        print("🚀 Starting STRICT Vietnamese Emoticon/Kaomoji Data Pipeline...")
        print("🚫 NO EMOJI IMAGES - TEXT EMOTICONS ONLY")
        print("="*70)
        
        try:
            self.step1_collect_raw_data()
            self.step2_strict_emoticon_filtering()
            self.step3_remove_duplicates()
            self.step4_analyze_data()
            self.save_results()
            
            print("\n" + "="*70)
            print("🎉 STRICT PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"✅ Final dataset: {len(self.final_data):,} unique TEXT emoticon/kaomoji samples")
            print(f"✅ Training-ready: {self.analysis_results['labeled_samples']:,} labeled samples")
            print("🚫 Zero emoji images included")
            print("📁 Results saved in ../data/ directory")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    pipeline = StrictVietnameseEmoticonPipeline()
    pipeline.run_complete_pipeline()
