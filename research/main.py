#!/usr/bin/env python3
"""
Vietnamese Emoticon/Kaomoji Sentiment Analysis Training
Using Emotiment Module for Training Vietnamese Sentiment Models
"""

import sys
from pathlib import Path

# Add emotiment module to path
sys.path.append(str(Path(__file__).parent))

from emotiment.training.trainer import Trainer
from emotiment.constant import GRID_MODE, MONO_MODE, MODEL_LIST
from emotiment.constant.training_constant import EMOJI2DESCRIPTION_METHOD


def demo_training():
    """Demo training with BiGRU model"""
    print("🎯 Vietnamese Emoticon/Kaomoji Sentiment Analysis")
    print("🚀 Demo Training with BiGRU Model")
    print("=" * 60)
    
    try:
        # Initialize trainer
        print("🔄 Initializing trainer...")
        trainer = Trainer(
            mode=MONO_MODE,
            method=EMOJI2DESCRIPTION_METHOD,
            head_name='bigru'
        )
        
        # Show dataset info
        if isinstance(trainer.input, tuple) and len(trainer.input) == 3:
            train_data, eval_data, test_data = trainer.input
            print(f"📊 Dataset loaded successfully!")
            print(f"   📚 Train set: {len(train_data)} samples")
            print(f"   📝 Eval set: {len(eval_data)} samples")  
            print(f"   🧪 Test set: {len(test_data)} samples")
            print(f"   📈 Total: {len(train_data) + len(eval_data) + len(test_data)} samples")
        
        print(f"🤖 Model: BiGRU (Bidirectional GRU)")
        print(f"📁 Output: {trainer.trainer.output_root}")
        print("")
        
        # Start training
        print("🚀 Starting training...")
        trainer.train()
        print("✅ Training completed!")
        
        # Evaluate
        print("\n📊 Starting evaluation...")
        results = trainer.evaluate()
        
        print("\n🎉 Training Results:")
        print("=" * 40)
        if isinstance(results, dict):
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    print(f"📈 {metric}: {value:.4f}")
                else:
                    print(f"📈 {metric}: {value}")
        else:
            print(f"📈 Results: {results}")
        
        print("\n✅ Vietnamese Emoticon/Kaomoji sentiment model trained successfully!")
        print("🎯 Model ready for inference on Vietnamese text with emoticons")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_all_models():
    """Train all available models in grid mode"""
    print("🎯 Vietnamese Emoticon/Kaomoji Sentiment Analysis")  
    print("🚀 Grid Training - All Models")
    print("=" * 60)
    
    try:
        # Initialize trainer for all models
        print("🔄 Initializing grid trainer...")
        trainer = Trainer(
            mode=GRID_MODE,
            method=EMOJI2DESCRIPTION_METHOD
        )
        
        # Show dataset info
        if isinstance(trainer.input, tuple) and len(trainer.input) == 3:
            train_data, eval_data, test_data = trainer.input
            print(f"📊 Dataset loaded successfully!")
            print(f"   📚 Train set: {len(train_data)} samples")
            print(f"   📝 Eval set: {len(eval_data)} samples")
            print(f"   🧪 Test set: {len(test_data)} samples")
        
        print(f"🤖 Training models: {MODEL_LIST}")
        print(f"📁 Output: {trainer.trainer.output_root}")
        print("")
        
        # Train all models
        print("🚀 Starting grid training...")
        trainer.train()
        print("✅ All models trained!")
        
        # Evaluate all models
        print("\n📊 Evaluating all models...")
        results = trainer.evaluate()
        
        print("\n🏆 Grid Training Results:")
        print("=" * 50)
        if isinstance(results, dict):
            for model_name, metrics in results.items():
                print(f"\n🤖 {model_name.upper()}:")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   📈 {metric}: {value:.4f}")
                        else:
                            print(f"   📈 {metric}: {value}")
                else:
                    print(f"   📈 Result: {metrics}")
        
        return results
        
    except Exception as e:
        print(f"❌ Grid training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function with interactive menu"""
    print("🇻🇳 Vietnamese Emoticon/Kaomoji Sentiment Analysis")
    print("=" * 60)
    print("Choose training mode:")
    print("1. 🎮 Demo Training (BiGRU model only)")
    print("2. 🔥 Train All Models (Grid search)")
    print("3. 🚪 Exit")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            return demo_training()
        elif choice == "2":
            return train_all_models()
        elif choice == "3":
            print("👋 Goodbye!")
            return None
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
            return main()
            
    except KeyboardInterrupt:
        print("\n👋 Training cancelled by user.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_training()
        elif sys.argv[1] == "--grid":  
            train_all_models()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python main.py           - Interactive mode")
            print("  python main.py --demo    - Demo training (BiGRU only)")
            print("  python main.py --grid    - Train all models")
            print("  python main.py --help    - Show this help")
        else:
            print(f"❌ Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        main()
