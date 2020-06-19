#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -n num_train_samples -i input_file -d data_file\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-n Number of training samples\n"
    printf "\t-i Workflow input file\n"
    printf "\t-d Data file path\n"
    printf "\t-o Workflow output dir\n"
    exit 1
}

# Argument parsing
while getopts "p:n:i:d:o:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        n ) NUM_SAMPLES="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        d ) DATA_FILE="$OPTARG" ;;
        o ) OUTPUT_DIR="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${NUM_SAMPLES}" ] || [ -z "${INPUT_FILE}" ] || \
    [ -z "${DATA_FILE}" ] || [ -z "${OUTPUT_DIR}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
python3 "${PROJECT_PATH}/code/configurate_ml.py" "${NUM_SAMPLES}" "${DATA_FILE}" "${INPUT_FILE}" "${OUTPUT_DIR}"
