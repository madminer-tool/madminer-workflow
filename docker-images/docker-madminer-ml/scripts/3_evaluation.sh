#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file -m model_file -d data_file\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-m Compressed model path\n"
    printf "\t-d Data file path\n"
    exit 1
}

# Argument parsing
while getopts "p:i:m:d:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        m ) MODEL_FILE="$OPTARG" ;;
        d ) DATA_FILE="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${MODEL_FILE}" ] || [ -z "${DATA_FILE}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Define auxiliary variables
MODELS_ABS_PATH="${PROJECT_PATH}/models"
RATES_ABS_PATH="${PROJECT_PATH}/rates"
RESULTS_ABS_PATH="${PROJECT_PATH}/results"
TESTS_ABS_PATH="${PROJECT_PATH}/test"


# Cleanup previous files (useful when run locally)
rm -rf "${RESULTS_ABS_PATH}"
rm -rf "${TESTS_ABS_PATH}"

mkdir -p "${MODELS_ABS_PATH}"
mkdir -p "${RATES_ABS_PATH}"
mkdir -p "${RESULTS_ABS_PATH}"
mkdir -p "${TESTS_ABS_PATH}"


# Unzip the models folder contents to identify the model
tar -xvf "${MODEL_FILE}" -C "${MODELS_ABS_PATH}"

MODEL_NAME=$(find "${MODELS_ABS_PATH}" -type d -mindepth 1 -maxdepth 1 -exec basename {} \;)
MODEL_DIR="${MODELS_ABS_PATH}/${MODEL_NAME}"


python3 "${PROJECT_PATH}/code/evaluation.py" "${INPUT_FILE}" "${MODEL_DIR}" "${DATA_FILE}"

tar -czvf "${PROJECT_PATH}/Results_${MODEL_NAME}.tar.gz" \
    -C "${PROJECT_PATH}" \
    "models" \
    "rates" \
    "results" \
    "test"

# cp -R /madminer/Results.tar.gz  {evalworkdir}
# ls -lR
