from clearml import Dataset

#### PARAMS ####

DATASET_PROJECT = 'datasets/speech_translation'
DATASET_NAME = 'id_data_2gb'
PARENT_DATASET_ID = '9904e5f0fe2e4ddcb56defe61d3c1b0a'
LOCAL_DATASET_DIR = '/home/digitalhub/Desktop/projects/datasets/st_id_data_2gb'

# s3://<server url>:<port>/<bucket>/...
OUTPUT_URI = 's3://experiment-logging/storage/datasets'

################

# gets the state from the parent dataset, which contains metadata of what files are uploaded
dataset = Dataset.create(
    dataset_project = DATASET_PROJECT,
    dataset_name = DATASET_NAME,
    parent_datasets = [PARENT_DATASET_ID,],
    )
# add files in specified folder, it cross-references to the current state, and ensures only new data is added
dataset.add_files(LOCAL_DATASET_DIR)

# upload dataset to remote storage
dataset.upload(
    output_url=OUTPUT_URI, 
    verbose=True
    )

dataset.finalize()