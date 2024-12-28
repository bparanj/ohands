from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from typing import Dict, List
import pandas as pd
from datetime import datetime

class ModelExplorer:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.results = []
        
    def load_model(self, model_name: str) -> None:
        """Load a model and its tokenizer"""
        print(f"\nLoading {model_name}...")
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load model with lower precision to save memory
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            
        # Force garbage collection
        import gc
        gc.collect()
    
    def generate_text(self, 
                     model_name: str, 
                     prompt: str, 
                     max_length: int = 100,
                     num_samples: int = 1,
                     temperature: float = 0.7,
                     top_k: int = 50,
                     top_p: float = 0.9) -> List[str]:
        """Generate text using the specified model"""
        if model_name not in self.models:
            print(f"Model {model_name} not loaded")
            return []
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Record start time
        start_time = time.time()
        
        # Generate text with memory optimization
        with torch.no_grad():  # Disable gradient calculation
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Record end time
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Store results
        self.results.append({
            'model': model_name,
            'prompt': prompt,
            'generated_text': generated_texts[0],  # Store first generation
            'generation_time': generation_time,
            'num_tokens': len(inputs[0]),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return generated_texts
    
    def compare_models(self, 
                      prompts: List[str],
                      model_names: List[str] = None) -> pd.DataFrame:
        """Compare text generation across different models"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = []
        for prompt in prompts:
            for model_name in model_names:
                print(f"\nGenerating text with {model_name}")
                print(f"Prompt: {prompt}")
                
                generated_texts = self.generate_text(model_name, prompt)
                if generated_texts:
                    print(f"Generated text: {generated_texts[0]}\n")
        
        return pd.DataFrame(self.results)
    
    def get_model_stats(self) -> Dict:
        """Get statistics about loaded models"""
        stats = {}
        for model_name, model in self.models.items():
            stats[model_name] = {
                'parameters': sum(p.numel() for p in model.parameters()),
                'layers': len(list(model.modules())),
                'vocab_size': self.tokenizers[model_name].vocab_size
            }
        return stats

def demonstrate_model_variants():
    # Initialize model explorer
    explorer = ModelExplorer()
    
    # Define model variants to compare (using smaller models)
    model_variants = [
        'distilgpt2',     # 82M parameters
        'gpt2'            # 124M parameters
    ]
    
    # Load models
    for model_name in model_variants:
        explorer.load_model(model_name)
    
    # Define prompts for comparison
    prompts = [
        "The future of artificial intelligence will",
        "Once upon a time in a digital world",
        "The solution to climate change requires"
    ]
    
    # Compare models
    print("\nComparing model variants...")
    results_df = explorer.compare_models(prompts)
    
    # Get model statistics
    print("\nModel Statistics:")
    stats = explorer.get_model_stats()
    for model_name, model_stats in stats.items():
        print(f"\n{model_name}:")
        print(f"Parameters: {model_stats['parameters']:,}")
        print(f"Layers: {model_stats['layers']}")
        print(f"Vocabulary Size: {model_stats['vocab_size']:,}")
    
    # Analysis of results
    if not results_df.empty:
        print("\nGeneration Time Analysis:")
        avg_times = results_df.groupby('model')['generation_time'].mean()
        print(avg_times)
        
        # Save results to CSV
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("\nResults saved to model_comparison_results.csv")
    else:
        print("\nNo results to analyze. Check if models were loaded successfully.")

if __name__ == "__main__":
    print("Model Variants Exploration")
    print("=" * 50)
    demonstrate_model_variants()