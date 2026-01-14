# Music style transfer

This repository contains the implementation of a music style transfer model using Variational Autoencoders (VAE). The model is designed to transfer the style of one piece of music to another while preserving the content.

## Project Structure

- `model/`: Training scripts and model definitions.
- `preprocessing/`: Data preprocessing utilities.
- `data/`: Datasets and analysis results.
- `evaluation/`: Evaluation scripts and metrics.
- `utils/`: Helper functions for audio, files, and plotting.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `docs/`: Documentation, thesis, and diagrams.
- `pipeline_tests/`: Unit and integration tests.

## Methodology & Results

- Master's thesis (Spanish): [docs/Tesis.pdf](https://github.com/LucasSom/style-transfer/blob/main/docs/Tesis.pdf)
- Paper (English): Somacal, L., Riera, P., Fernández Slezak, D. and Miguel, M. "Symbolic music style transfer via latent space transformations. Model and evaluation", published in [Proceedings of the 1st Latin American Music Information Retrieval Workshop (LAMIR), 2024](https://zenodo.org/records/14908040)

The VAE implementation is based on [Rui Guo's work](https://github.com/ruiguo-bio/colab_tension_vae.git).

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
sh dependencies.sh
```

## Usage

The code is based on modular scripts for training, inference, and evaluation.
The project uses [pydoit](https://pydoit.org/) as a task runner to automate the machine learning pipeline.
All tasks are defined in `dodo.py`.

### Running Tasks
Execute tasks using the `doit` command:

```bash
# List all available tasks
doit list

# Run a specific task
doit <task_name>

# Run a task with a specific subtask
doit <task_name>:<subtask_name>

# Run multiple tasks
doit <task_1> <task_2>

# Show task details
doit info <task_name>
```

### Available Tasks
#### Data Preparation
* `preprocess`: Preprocesses MIDI datasets into piano rolls, considering subdatasets (Bach, Mozart, Frescobaldi, ragtime, Lakh MIDI).
* `split_dataset`: Splits preprocessed data into train/test/validation sets using stratified sampling.
* `assemble_data_to_analyze`: Prepares data for statistical analysis, including musicality distributions.
* `oversample`: Balances minority classes in the training set.

#### Data Analysis
* `analyze_data`: Performs various dataset analyses:
  * `style_closeness`: Analyzes style similarities
  * `distances_distribution`: Computes optimal transport distances between styles
  * `musicality`: Evaluates melodic and rhythmic distributions
  * `entropies`: Calculates bigram entropies per style
  * `style_histograms`: Generates style-specific feature histograms
  * `confusion_matrix`: Creates confusion matrices for style classification
  * `style_differences`: Visualizes differences between styles

#### Model Training & Testing
* `train`: Trains the VAE model on the preprocessed dataset.
* `test`: Generates t-SNE visualizations of the latent space and interpolations between style centroids.
* `plot_training_metrics`: Plots training loss curves and metrics.

#### Embeddings & Style Transfer
* `embeddings`: Computes latent embeddings for songs and style characteristics.
* `reconstruct`: Generates reconstructions from the learned embeddings.
* `transfer_style`: Performs style transfer between different musical styles using various mutation operators and alpha values.
* `transfer_new_rolls`: Transfers style on new, external MIDI files.

#### Evaluation & Visualization
* `metrics`: Calculates evaluation metrics (plagiarism, intervals, rhythmic bigrams).
* `evaluation`: Evaluates style transfer quality using multiple criteria.
* `overall_single_evaluation`: Aggregates metrics across all style transfers for a single model.
* `family_evaluation`: Compares metrics across model families (e.g., fine-tuned vs. pre-trained).
* `all_models_evaluation`: Generates comparative boxplots for all model families.
* `sample_audios`: Generates audio files (MP3/MIDI) from transferred styles.
* `sample_sheets`: Creates music notation sheets for selected examples.
* `html`: Builds an interactive HTML interface to browse and compare results.
* `examples`: Generates curated examples for each style transfer direction.

### Example Workflow

```bash
# 1. Preprocess data
doit preprocess

# 2. Split into train/test/val
doit split_dataset

# 3. Balance classes
doit oversample

# 4. Train model
doit train:4-Lakh_Kern-96

# 5. Generate embeddings
doit embeddings:4-Lakh_Kern-96

# 6. Reconstruct and transfer styles
doit reconstruct:4-Lakh_Kern-96
doit transfer_style:4-Lakh_Kern-96

# 7. Evaluate results
doit metrics:4-Lakh_Kern-96
doit evaluation:4-Lakh_Kern-96

# 8. Generate outputs
doit sample_audios:4-Lakh_Kern-96
doit html:4-Lakh_Kern-96
```

Tasks have dependencies and will automatically run prerequisite tasks when needed.
