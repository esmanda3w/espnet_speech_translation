# Speech Translation with ESPnet
This repository consists of two translation pipelines using the two approaches for speech translation provided in ESPnet. The first approach is Cascade-ST, where Automatic Speech Recognition (ASR) modules are cascaded into Machine Translation (MT) modules. The second approach is end-to-end speech translation (E2E-ST), where source speech is directly mapped to its text translation in the target language.

## Project Organization
The directory structure of this project is as follows.

```
    ├── docker-compose.yml
    ├── Dockerfile
    ├── README.md
    ├── models/      # contains pretrained models
    ├── mount/       # contains the recipes to run the different pipelines
    │   └── cascade/         # Cascade-ST recipe
    │   └── e2e_st/          # E2E-ST recipe
    │   └── java/            # ASR recipe (using a pretrained model trained on Javanese)
    │   └── mt_test/         # MT recipe
    ├── nltk_data/   # pre-downloaded nltk packages to prevent crash when downloading in pipeline
    └── scripts/     # scripts used across different recipes

```

## Getting Started

### Choice 1: Pull the docker image from Docker Hub
Pull the prebuilt docker image that was uploaded onto Docker Hub (special thanks to my mentor Daniel).
```bash
docker pull dleongsh/espnet:202205-torch1.10-cu113-runtime
# if you want to use tensorboard
docker pull dleongsh/tensorboard:latest
```

### Choice 2: Build the docker image
If the prebuilt image doesn't work, build the image yourself using the following command.
```
docker build -t dleongsh/espnet:202205-torch1.10-cu113-runtime .
```

## Prep work for all recipes
### Data preparation
The data preparation script for each recipe is found under `mount/<recipe>/local/data.sh`. The script assumes that the input data folder follows the file structure as outputted from the `JTubeSpeech` repository that I forked. 

Within each recipe, point the `train_data_folder` and `test_data_folder` to the correct train and test dataset folder names.

### Storage of logs
All the logs are redirected to the `mount/exp` folder. The structure of the directory is as follows.
```
├── mount/exp
│   └── <main folder>   # name of recipe being run (e.g. java, mt_test, e2e_st, etc)
│   └── <sub folder>    # name of the experiment
```
To change where the experiment logs are stored, change the following parameters: 
- `--<task>_exp`           # task: asr/mt/st
- `--<task>_stats_dir`
- `--lm_exp`
- `--lm_stats_dir`

### Changing model configurations
Model specific configurations are stored in the `conf/` folder and can be modified.

#### Package trained model (optional for ASR and MT recipes only)
This step is done only if you want to use the trained model for the Cascade-ST recipe. 

Set the `package_model` flag to `true` and provide a name for the packaged model in the `package_model_name` option. The packaged model will then be stored in the `models/` folder, according to the directory structure expected by the Cascade-ST recipe.

## Running Cascade-ST pipeline

### Step 1: Run ASR recipe (using a pretrained model trained on Javanese)

#### Pretrained model setup
Download the [jv_openslr35 pretrained model](https://zenodo.org/record/5090139#.YuI8EDlBz8k). Recreate the directory structure below (you will only need a few files from the downloaded zip).
```
├── models/
│   └── jv_openslr35/
│   │   └── bpe_unigram1000/
│   │   │   └── bpe.model
│   │   └── config.yaml
│   │   └── valid.acc.best.pth

```

#### Running the recipe
Go to the `docker-compose.yaml` file and uncomment the line `command: bash -c "cd /mount/java/asr1 && ./run.sh"`. Then run the docker-compose file as described [here](#running-the-docker-compose).


### Step 2: Run MT recipe

#### Running the recipe
Go to the `docker-compose.yaml` file and uncomment the line `command: bash -c "cd /mount/mt_test/mt1 && ./run.sh"`. Then run the docker-compose file as described [here](#running-the-docker-compose).


### Step 3: Run Cascade-ST recipe

#### Using the trained ASR model
Specify the name of the ASR main folder under the `asr_main_folder` option and the package name in `asr_package_model_name`.

Change the following configurations according to your trained ASR model under the *setting up asr* segment:
- `bpe_model`
- `inference_asr_model`
- `use_lm`
- `lm_model`

If you already ran ASR inference while running the ASR recipe and want to direct the output for MT inference, you can skip re-running of the ASR inference by specifying `true` for the `skip_asr` flag. You must also change the `asr_inference_text` option under the MT section to point to where the asr inference text is (e.g. `mount/exp/<asr_main_folder>/<asr_sub_folder>/asr_exp/*/*/text`).

#### Using the trained MT model
Specify the name of the MT main folder under the `mt_main_folder` option and the package name in `mt_package_model_name`.

Change the following configurations according to your trained MT model under the *setting up mt* segment:
- `inference_mt_model`

#### Running the recipe
Go to the `docker-compose.yaml` file and uncomment the line `command: bash -c "cd /mount/cascade && ./run.sh"`. Then run the docker-compose file as described [here](#running-the-docker-compose).

## Running E2E-ST pipeline

### Initialising ASR encoder and MT decoder
To initialise the E2E-ST model with a trained ASR encoder and MT decoder, specify the path to the respective trained models under the `st_args` option.

### Running locally
Go to the `docker-compose.yaml` file and uncomment the line `command: bash -c "cd /mount/e2e_st/st1 && ./run.sh --local_run true`. Then run the docker-compose file as described [here](#running-the-docker-compose).

### Running on ClearML
Run the `mount/e2e_st/st1/run.py` file using `python3 run.py`.

## Running the docker-compose

1. Look for the line under volumes `/home/digitalhub/Desktop/projects/datasets:/datasets` and change the mounting path to where your data directory is stored locally.

2. Uncomment the command to be run depending on which recipe you want to run.

3. Run the following command.
```bash
docker-compose up
```

4. To view the tensorboard, go to [localhost:6006](localhost:6006) on your browser.
