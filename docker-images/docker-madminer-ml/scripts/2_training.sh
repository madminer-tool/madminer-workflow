#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file -t train_folder\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-t Regex selecting the train folders\n"
    exit 1
}

# Argument parsing
while getopts "p:i:t:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        t ) TRAIN_FOLDER="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${TRAIN_FOLDER}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Define auxiliary variables
MODEL_FILE_ABS_PATH="${PROJECT_PATH}/Model.tar.gz"
MODEL_INFO_ABS_PATH="${PROJECT_PATH}/models"
SAMPLES_ABS_PATH="${PROJECT_PATH}/data/${TRAIN_FOLDER}"


# Cleanup previous files (useful when run locally)
rm -rf "${MODEL_FILE_ABS_PATH}"
rm -rf "${MODEL_INFO_ABS_PATH}"

mkdir -p "${MODEL_INFO_ABS_PATH}"


# Perform actions
python3 "${PROJECT_PATH}/code/train.py" "${SAMPLES_ABS_PATH}" "${INPUT_FILE}"
tar -czvf "${MODEL_FILE_ABS_PATH}" -C "${MODEL_INFO_ABS_PATH}" .

# cp -R /madminer/Model.tar.gz  {trainworkdir}
