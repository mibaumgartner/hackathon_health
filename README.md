AI-HERO Health Challenge on energy efficient AI - Predicting Covid-19 from chest x-ray images
=============================================================================================

Prototyping repository for our solution to the AI-HERO Health Challenge Hackathon.

### Setup Env

Run following commands from the root dir of this repository (tested with python=3.8):

```bash
pip install -r requirements.txt
pip install .
```

### Running Inference

Inference can be run with the following command:

```bash
sbatch eval_conda.sh --save_dir [dir to save results] --csv_filepath [Filepath of csv] --data_dir [directory containing imgs of csv file] --ckpt_path [path to checkpoint]
```

Details:

1. _save_dir:_ Directory where predictions will be saved to as 'predictions.csv'
2. _data_dir:_ Directory to the folder containing the images of the csv file.
3. _csv_filepath:_ Path to the csv file containing the "images" in the first column and the first row is the header
4. _ckpt_path:_ Path to the checkpoint file loaded from Zenodo.
