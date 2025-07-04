import os
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from collections import Counter
import warnings
import glob

warnings.filterwarnings('ignore')


class ISLDataValidator :
    def __init__(self, csv_path, frames_base_path, output_dir="validation_output") :
        self.csv_path = csv_path
        self.frames_base_path = frames_base_path
        self.output_dir = output_dir
        self.validation_results = {}
        self.frame_mapping = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def discover_frame_structure(self) :
        """Discover the frame directory structure"""
        print("=== Discovering Frame Structure ===")

        # Look for common frame directory patterns
        frame_dirs = []
        patterns = [
            os.path.join(self.frames_base_path, "**", "*.jpg"),
            os.path.join(self.frames_base_path, "**", "*.png"),
            os.path.join(self.frames_base_path, "**", "*.jpeg"),
            os.path.join(self.frames_base_path, "frames", "**", "*.jpg"),
            os.path.join(self.frames_base_path, "images", "**", "*.jpg"),
        ]

        all_frame_files = []
        for pattern in patterns :
            files = glob.glob(pattern, recursive=True)
            all_frame_files.extend(files)

        if not all_frame_files :
            print("⚠ No frame files found with common extensions (.jpg, .png, .jpeg)")
            return {}

        print(f"✓ Found {len(all_frame_files)} frame files")

        # Group by directory
        dir_structure = {}
        for file_path in all_frame_files :
            dir_path = os.path.dirname(file_path)
            rel_dir = os.path.relpath(dir_path, self.frames_base_path)
            if rel_dir not in dir_structure :
                dir_structure[rel_dir] = []
            dir_structure[rel_dir].append(file_path)

        print(f"✓ Found {len(dir_structure)} directories with frames")
        print("Sample directories:")
        for i, (dir_name, files) in enumerate(list(dir_structure.items())[:5]) :
            print(f"  {i + 1}. {dir_name} ({len(files)} files)")

        return dir_structure

    def validate_csv_structure(self) :
        """Validate CSV file structure and content"""
        print("=== CSV Validation ===")

        try :
            df = pd.read_csv(self.csv_path)
            print(f"✓ CSV loaded successfully")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {df.columns.tolist()}")

            # Store basic info
            self.validation_results['csv_info'] = {
                'shape' : df.shape,
                'columns' : df.columns.tolist(),
                'memory_usage_mb' : df.memory_usage(deep=True).sum() / 1024 ** 2
            }

            # Check for required columns (flexible detection)
            possible_sentence_cols = ['Sentence', 'sentence', 'label', 'Label', 'text', 'Text']
            possible_gloss_cols = ['SIGN GLOSSES', 'sign_glosses', 'glosses', 'Glosses', 'signs', 'Signs']

            sentence_col = None
            gloss_col = None

            for col in possible_sentence_cols :
                if col in df.columns :
                    sentence_col = col
                    break

            for col in possible_gloss_cols :
                if col in df.columns :
                    gloss_col = col
                    break

            if not sentence_col :
                print(f"✗ No sentence column found. Expected one of: {possible_sentence_cols}")
                return None

            if not gloss_col :
                print(f"⚠ No sign glosses column found. Expected one of: {possible_gloss_cols}")
                print("  Will proceed with sentence column only")

            print(f"✓ Using sentence column: '{sentence_col}'")
            if gloss_col :
                print(f"✓ Using sign glosses column: '{gloss_col}'")

            # Standardize column names
            df = df.rename(columns={sentence_col : 'Sentence'})
            if gloss_col :
                df = df.rename(columns={gloss_col : 'SIGN_GLOSSES'})

            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0 :
                print(f"⚠ Missing values found:")
                for col, count in missing_values.items() :
                    if count > 0 :
                        print(f"    {col}: {count} missing values ({count / len(df) * 100:.1f}%)")

                # Remove rows with missing critical values
                before_len = len(df)
                df = df.dropna(subset=['Sentence'])
                after_len = len(df)
                if before_len != after_len :
                    print(f"  Removed {before_len - after_len} rows with missing sentences")
            else :
                print(f"✓ No missing values")

            # Clean and validate sentences
            df['Sentence'] = df['Sentence'].astype(str).str.strip()
            df = df[df['Sentence'] != '']  # Remove empty sentences

            # Analyze sentences
            unique_sentences = df['Sentence'].unique()
            print(f"✓ Found {len(unique_sentences)} unique sentences")

            # Show sentence distribution
            sentence_counts = df['Sentence'].value_counts()
            print(f"  Sentence occurrences:")
            print(f"    Mean: {sentence_counts.mean():.1f}")
            print(f"    Min: {sentence_counts.min()}")
            print(f"    Max: {sentence_counts.max()}")
            print(f"    Median: {sentence_counts.median():.1f}")

            # Store sentence statistics
            self.validation_results['sentence_stats'] = {
                'unique_sentences' : len(unique_sentences),
                'total_entries' : len(df),
                'avg_occurrences' : float(sentence_counts.mean()),
                'min_occurrences' : int(sentence_counts.min()),
                'max_occurrences' : int(sentence_counts.max()),
                'median_occurrences' : float(sentence_counts.median())
            }

            # Analyze sign glosses if available
            if 'SIGN_GLOSSES' in df.columns :
                self._analyze_sign_glosses(df)

            # Check for potential issues
            self._check_data_quality_issues(df)

            return df

        except Exception as e :
            print(f"✗ Error loading CSV: {e}")
            return None

    def _analyze_sign_glosses(self, df) :
        """Analyze sign glosses column"""
        print(f"\n--- Sign Glosses Analysis ---")

        # Clean glosses
        df['SIGN_GLOSSES'] = df['SIGN_GLOSSES'].fillna('').astype(str).str.strip()

        # Count glosses
        all_glosses = []
        gloss_lengths = []

        for glosses in df['SIGN_GLOSSES'] :
            if glosses and glosses != '' :
                # Split by common delimiters
                gloss_list = [g.strip() for g in glosses.replace(',', ' ').split() if g.strip()]
                all_glosses.extend(gloss_list)
                gloss_lengths.append(len(gloss_list))
            else :
                gloss_lengths.append(0)

        if all_glosses :
            gloss_counter = Counter(all_glosses)
            print(f"✓ Found {len(gloss_counter)} unique sign glosses")
            print(f"  Total gloss tokens: {len(all_glosses)}")
            print(f"  Average glosses per sentence: {np.mean(gloss_lengths):.1f}")
            print(f"  Gloss sequence lengths: {min(gloss_lengths)}-{max(gloss_lengths)}")

            # Show top glosses
            print(f"  Top 10 glosses: {list(gloss_counter.most_common(10))}")

            # Store gloss statistics
            self.validation_results['gloss_stats'] = {
                'unique_glosses' : len(gloss_counter),
                'total_tokens' : len(all_glosses),
                'avg_per_sentence' : float(np.mean(gloss_lengths)),
                'length_range' : [int(min(gloss_lengths)), int(max(gloss_lengths))],
                'top_glosses' : dict(gloss_counter.most_common(20))
            }
        else :
            print(f"⚠ No valid sign glosses found")

    def _check_data_quality_issues(self, df) :
        """Check for common data quality issues"""
        print(f"\n--- Data Quality Checks ---")

        issues = []

        # Check for duplicate sentences
        duplicate_sentences = df['Sentence'].duplicated().sum()
        if duplicate_sentences > 0 :
            issues.append(f"Found {duplicate_sentences} duplicate sentences")
            print(f"⚠ Found {duplicate_sentences} duplicate sentences")

        # Check for very short sentences
        short_sentences = df[df['Sentence'].str.len() < 5]
        if len(short_sentences) > 0 :
            issues.append(f"Found {len(short_sentences)} very short sentences")
            print(f"⚠ {len(short_sentences)} sentences are very short (<5 chars)")

        # Check for very long sentences
        long_sentences = df[df['Sentence'].str.len() > 100]
        if len(long_sentences) > 0 :
            issues.append(f"Found {len(long_sentences)} very long sentences")
            print(f"⚠ {len(long_sentences)} sentences are very long (>100 chars)")

        self.validation_results['data_quality_issues'] = issues

        if not issues :
            print(f"✓ No major data quality issues found")

    def match_sentences_to_frames(self, df, dir_structure) :
        """Attempt to match sentences to frame directories"""
        print("\n=== Matching Sentences to Frames ===")

        if not dir_structure :
            print("⚠ No frame directories found to match")
            return df

        # Create a mapping based on sentence similarity or directory names
        matched_paths = []
        unmatched_sentences = []

        for idx, row in df.iterrows() :
            sentence = row['Sentence']
            matched_dir = None

            # Try to find matching directory based on sentence content
            # This is a simple heuristic - you may need to customize this
            for dir_name, files in dir_structure.items() :
                # Check if sentence words appear in directory name
                sentence_words = sentence.lower().split()
                dir_words = dir_name.lower().replace('_', ' ').replace('-', ' ').split()

                # Simple matching heuristic
                if any(word in dir_name.lower() for word in sentence_words if len(word) > 3) :
                    matched_dir = dir_name
                    break

            if matched_dir :
                # Pick a representative frame from the directory
                frame_files = dir_structure[matched_dir]
                if frame_files :
                    matched_paths.append(frame_files[0])  # Use first frame as representative
                else :
                    matched_paths.append(None)
                    unmatched_sentences.append(sentence)
            else :
                matched_paths.append(None)
                unmatched_sentences.append(sentence)

        # Add paths to dataframe
        df['Frame_Path'] = matched_paths

        matched_count = sum(1 for path in matched_paths if path is not None)
        print(f"✓ Matched {matched_count}/{len(df)} sentences to frame directories")

        if unmatched_sentences :
            print(f"⚠ {len(unmatched_sentences)} sentences could not be matched to frames")
            print("  Sample unmatched sentences:")
            for i, sentence in enumerate(unmatched_sentences[:5]) :
                print(f"    {i + 1}. {sentence[:50]}...")

        return df

    def validate_matched_frames(self, df, sample_size=50) :
        """Validate matched frame paths"""
        print("\n=== Frame Validation ===")

        if 'Frame_Path' not in df.columns :
            print("⚠ No frame paths to validate")
            return False

        # Filter out None paths
        valid_paths = df[df['Frame_Path'].notna()]

        if len(valid_paths) == 0 :
            print("⚠ No valid frame paths found")
            return False

        # Sample for validation
        sample_paths = valid_paths.sample(min(sample_size, len(valid_paths)))['Frame_Path'].tolist()

        print(f"Validating {len(sample_paths)} frame paths...")

        valid_count = 0
        image_info = {'dimensions' : [], 'sizes' : [], 'formats' : []}

        for path in sample_paths :
            if os.path.exists(path) :
                try :
                    img = cv2.imread(path)
                    if img is not None :
                        valid_count += 1
                        image_info['dimensions'].append(img.shape[:2])
                        image_info['sizes'].append(os.path.getsize(path))

                        _, ext = os.path.splitext(path)
                        image_info['formats'].append(ext.lower())
                except Exception as e :
                    print(f"  Error reading {os.path.basename(path)}: {str(e)[:50]}")

        validity_rate = valid_count / len(sample_paths)
        print(f"✓ Valid frames: {valid_count}/{len(sample_paths)} ({validity_rate:.1%})")

        if image_info['dimensions'] :
            dimensions = np.array(image_info['dimensions'])
            sizes = np.array(image_info['sizes'])
            formats = Counter(image_info['formats'])

            print(f"  Image properties:")
            print(f"    Dimensions: {dimensions.min(axis=0)} to {dimensions.max(axis=0)}")
            print(f"    File sizes: {sizes.min() / 1024:.1f} - {sizes.max() / 1024:.1f} KB")
            print(f"    Formats: {dict(formats)}")

            # Store image statistics
            self.validation_results['image_stats'] = {
                'validity_rate' : float(validity_rate),
                'sample_size' : len(sample_paths),
                'matched_sentences' : len(valid_paths),
                'total_sentences' : len(df),
                'dimensions_range' : {
                    'min' : dimensions.min(axis=0).tolist(),
                    'max' : dimensions.max(axis=0).tolist(),
                    'median' : np.median(dimensions, axis=0).astype(int).tolist()
                },
                'formats' : dict(formats)
            }

        return validity_rate > 0.5

    def create_validation_plots(self, df) :
        """Create visualization plots"""
        print(f"\n=== Creating Validation Plots ===")

        save_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # Plot 1: Sentence analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ISL Dataset Analysis', fontsize=16)

        # Sentence length distribution
        sentence_lengths = df['Sentence'].str.len()
        axes[0, 0].hist(sentence_lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Sentence Length Distribution (characters)')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        # Word count distribution
        word_counts = df['Sentence'].str.split().str.len()
        axes[0, 1].hist(word_counts, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Sentence Length Distribution (words)')
        axes[0, 1].set_xlabel('Words')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Sentence frequency
        sentence_counts = df['Sentence'].value_counts()
        axes[1, 0].hist(sentence_counts.values, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Sentence Frequency Distribution')
        axes[1, 0].set_xlabel('Occurrences')
        axes[1, 0].set_ylabel('Number of Sentences')
        axes[1, 0].grid(True, alpha=0.3)

        # Frame matching status
        if 'Frame_Path' in df.columns :
            matched = df['Frame_Path'].notna().sum()
            unmatched = df['Frame_Path'].isna().sum()
            axes[1, 1].pie([matched, unmatched], labels=['Matched', 'Unmatched'],
                           autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Frame Matching Status')
        else :
            axes[1, 1].text(0.5, 0.5, 'No frame\nmatching\nperformed',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Frame Matching Status')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'dataset_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Sign glosses analysis (if available)
        if 'gloss_stats' in self.validation_results :
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Sign Glosses Analysis', fontsize=16)

            # Top glosses
            top_glosses = list(self.validation_results['gloss_stats']['top_glosses'].items())[:15]
            if top_glosses :
                glosses, counts = zip(*top_glosses)
                axes[0].barh(range(len(glosses)), counts)
                axes[0].set_yticks(range(len(glosses)))
                axes[0].set_yticklabels(glosses)
                axes[0].set_title('Top 15 Sign Glosses')
                axes[0].set_xlabel('Frequency')
                axes[0].grid(True, alpha=0.3)

            # Gloss sequence lengths
            if 'SIGN_GLOSSES' in df.columns :
                gloss_lengths = []
                for glosses in df['SIGN_GLOSSES'] :
                    if glosses and str(glosses) != 'nan' :
                        gloss_list = str(glosses).replace(',', ' ').split()
                        gloss_lengths.append(len(gloss_list))
                    else :
                        gloss_lengths.append(0)

                axes[1].hist(gloss_lengths, bins=15, alpha=0.7, edgecolor='black')
                axes[1].set_title('Gloss Sequence Length Distribution')
                axes[1].set_xlabel('Number of Glosses')
                axes[1].set_ylabel('Frequency')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'glosses_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✓ Plots saved to {save_dir}/")

    def suggest_next_steps(self, df) :
        """Suggest next steps based on validation results"""
        print(f"\n=== Recommendations ===")

        recommendations = []

        # Frame matching recommendations
        if 'Frame_Path' in df.columns :
            matched_count = df['Frame_Path'].notna().sum()
            if matched_count < len(df) * 0.5 :
                recommendations.append("Manual frame-sentence mapping needed")
                print(f"⚠ Only {matched_count}/{len(df)} sentences matched to frames")
                print(f"  Consider creating a mapping CSV with columns: Sentence, Frame_Directory")

        # Dataset size recommendations
        if len(df) < 100 :
            recommendations.append("Small dataset - consider data augmentation")
            print(f"⚠ Small dataset ({len(df)} samples) - consider data augmentation")

        # Sign glosses recommendations
        if 'gloss_stats' in self.validation_results :
            gloss_stats = self.validation_results['gloss_stats']
            if gloss_stats['unique_glosses'] > 1000 :
                recommendations.append("Large vocabulary - consider gloss frequency filtering")
                print(f"⚠ Large gloss vocabulary ({gloss_stats['unique_glosses']} glosses)")
                print(f"  Consider filtering low-frequency glosses")

        # Data preprocessing recommendations
        print(f"\n📋 Preprocessing Recommendations:")
        print(f"  1. Create explicit frame-sentence mapping")
        print(f"  2. Organize frames by sentence/sequence")
        print(f"  3. Consider frame sampling strategy (e.g., every Nth frame)")
        print(f"  4. Implement data augmentation for small classes")

        return recommendations

    def generate_report(self) :
        """Generate validation report"""
        print(f"\n=== Generating Report ===")

        # Add metadata
        self.validation_results['metadata'] = {
            'validation_date' : datetime.now().isoformat(),
            'csv_path' : self.csv_path,
            'frames_base_path' : self.frames_base_path,
            'validator_version' : '2.1.0'
        }

        # Save JSON report
        report_path = os.path.join(self.output_dir, 'validation_report.json')
        with open(report_path, 'w') as f :
            json.dump(self.validation_results, f, indent=2)

        # Generate summary
        summary_path = os.path.join(self.output_dir, 'validation_summary.txt')
        with open(summary_path, 'w') as f :
            f.write("ISL Dataset Validation Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"CSV Path: {self.csv_path}\n")
            f.write(f"Frames Path: {self.frames_base_path}\n\n")

            if 'csv_info' in self.validation_results :
                csv_info = self.validation_results['csv_info']
                f.write(f"Dataset Overview:\n")
                f.write(f"  - Total entries: {csv_info['shape'][0]:,}\n")
                f.write(f"  - Columns: {csv_info['columns']}\n\n")

            if 'sentence_stats' in self.validation_results :
                stats = self.validation_results['sentence_stats']
                f.write(f"Sentence Statistics:\n")
                f.write(f"  - Unique sentences: {stats['unique_sentences']:,}\n")
                f.write(f"  - Total entries: {stats['total_entries']:,}\n\n")

            if 'gloss_stats' in self.validation_results :
                gloss_stats = self.validation_results['gloss_stats']
                f.write(f"Sign Glosses:\n")
                f.write(f"  - Unique glosses: {gloss_stats['unique_glosses']:,}\n")
                f.write(f"  - Total tokens: {gloss_stats['total_tokens']:,}\n\n")

        print(f"✓ Report saved to: {report_path}")
        print(f"✓ Summary saved to: {summary_path}")

    def run_validation(self) :
        """Run the complete validation process"""
        print("Starting ISL Dataset Validation (Sign Glosses Mode)...")
        print("=" * 60)

        start_time = datetime.now()

        try :
            # Step 1: Validate CSV
            df = self.validate_csv_structure()
            if df is None :
                return False

            # Step 2: Discover frame structure
            dir_structure = self.discover_frame_structure()

            # Step 3: Try to match sentences to frames
            if dir_structure :
                df = self.match_sentences_to_frames(df, dir_structure)

                # Step 4: Validate matched frames
                if 'Frame_Path' in df.columns :
                    self.validate_matched_frames(df)

            # Step 5: Create plots
            self.create_validation_plots(df)

            # Step 6: Suggest next steps
            recommendations = self.suggest_next_steps(df)

            # Step 7: Generate report
            self.generate_report()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"\n" + "=" * 60)
            print(f"✓ Validation completed in {duration:.1f} seconds")
            print(f"Results saved to: {self.output_dir}/")

            return True

        except Exception as e :
            print(f"\n✗ Validation failed: {e}")
            return False


def main() :
    """Main function for sign glosses dataset"""
    print("ISL Dataset Validator v2.1 - Sign Glosses Mode")
    print("=" * 50)

    # Update these paths for your setup
    csv_path = r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL_Corpus_sign_glosses.csv"
    frames_path = r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"

    # Check paths
    if not os.path.exists(csv_path) :
        print(f"✗ CSV file not found: {csv_path}")
        return

    if not os.path.exists(frames_path) :
        print(f"✗ Frames directory not found: {frames_path}")
        return

    print(f"✓ CSV file: {csv_path}")
    print(f"✓ Frames directory: {frames_path}")

    # Create validator
    output_dir = "validation_output_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    validator = ISLDataValidator(csv_path, frames_path, output_dir)

    # Run validation
    success = validator.run_validation()

    if success :
        print(f"\n🎉 Next Steps:")
        print(f"1. Review validation results in: {output_dir}/")
        print(f"2. Create proper sentence-to-frame mapping")
        print(f"3. Organize frame sequences by sentence")
        print(f"4. Implement frame sampling strategy")
        print(f"5. Set up your training pipeline")
    else :
        print(f"\n❌ Validation encountered issues")
        print(f"Check the output directory for details: {output_dir}/")


if __name__ == "__main__" :
    main()