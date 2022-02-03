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
sbatch eval_conda.sh --save_dir [dir to save results] --data_dir [directory with csv file and img dir]--ckpt_path [path to checkpoint]
```

Details:

1. _save_dir:_ Directory where predictions will be saved to as 'predictions.csv'
2. _data_dir:_ Directory that contains (a 'test.csv' that contains only an 'image' column;"
             "The corresponding images are located '{data_dir}/imgs'
3. _ckpt_path:_ Path to the checkpoint file loaded from Zenodo.