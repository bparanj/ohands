# Language Model Analysis and Visualization Demos

This repository contains three comprehensive demos for analyzing and comparing language models:

1. Model Analysis Metrics Demo
2. Generation Parameters Demo
3. Visualization Demo

## 1. Model Analysis Metrics Demo

### Purpose
Provides detailed analysis of language models across multiple dimensions:
- Architecture metrics
- Inference speed
- Text quality
- Memory usage

### Key Components
- `ModelAnalyzer` class for comprehensive model analysis
- Architecture analysis (parameters, layers, etc.)
- Performance metrics (inference speed, memory usage)
- Text quality metrics (lexical diversity, repetition)

### Output
- Detailed CSV/Excel reports
- Comparative analysis between models
- Performance benchmarks
- Memory usage statistics

## 2. Generation Parameters Demo

### Purpose
Explores how different generation parameters affect model output:
- Temperature variations
- Top-K and Top-P sampling
- Beam search configurations

### Key Components
- `GenerationConfig` dataclass for parameter management
- `GenerationExplorer` class for systematic parameter exploration
- Temperature exploration (creativity vs. consistency)
- Sampling strategy comparison
- Beam search analysis

### Output
- Generated text samples with different parameters
- Parameter impact analysis
- CSV and JSON results
- Comparative output analysis

## 3. Visualization Demo

### Purpose
Creates interactive visualizations of model behavior and performance:
- Model architecture comparisons
- Attention pattern visualization
- Token distribution analysis
- Generation metrics visualization

### Key Components
- `ModelVisualizer` class for creating visualizations
- Interactive Plotly graphs
- Attention pattern heatmaps
- Token distribution plots
- Performance metric visualizations

### Output
- HTML interactive visualizations
- Architecture comparison plots
- Attention pattern heatmaps
- Token distribution graphs
- Generation metrics plots

## Setup Instructions

### Requirements
```bash
pip install transformers torch pandas numpy nltk plotly seaborn matplotlib
```

### Running the Demos

1. Analysis Metrics:
```bash
python model_analysis_metrics.py
```

2. Generation Parameters:
```bash
python generation_parameters_demo.py
```

3. Visualizations:
```bash
python visualization_demo.py
```

## Understanding the Results

### Analysis Metrics
- Model architecture comparison
- Performance benchmarks
- Quality metrics
- Resource usage

### Generation Parameters
- Impact of temperature on creativity
- Sampling strategy effects
- Beam search influence
- Parameter optimization insights

### Visualizations
- Interactive model comparisons
- Attention pattern analysis
- Token distribution insights
- Performance visualization

## Best Practices

1. **Model Selection**
   - Start with smaller models
   - Consider resource constraints
   - Match model to use case

2. **Parameter Tuning**
   - Experiment with temperature
   - Balance quality and diversity
   - Optimize for specific tasks

3. **Visualization Analysis**
   - Compare multiple models
   - Analyze attention patterns
   - Study token distributions

## Notes

- Results may vary by hardware
- GPU acceleration recommended
- First run downloads models
- Interactive visualizations require web browser

## Extending the Demos

1. **Analysis Metrics**
   - Add custom metrics
   - Implement task-specific evaluations
   - Extend memory analysis

2. **Generation Parameters**
   - Add new parameter combinations
   - Implement custom sampling strategies
   - Create parameter optimization tools

3. **Visualizations**
   - Add new visualization types
   - Customize plot styles
   - Implement real-time visualization

## Troubleshooting

1. **Memory Issues**
   - Use smaller models
   - Reduce batch sizes
   - Enable gradient checkpointing

2. **Performance**
   - Enable GPU acceleration
   - Optimize parameter settings
   - Use appropriate batch sizes

3. **Visualization**
   - Check browser compatibility
   - Reduce data size if needed
   - Use appropriate plot types