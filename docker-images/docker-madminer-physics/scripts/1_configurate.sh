#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file -o output_dir\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-o Workflow output dir\n"
    exit 1
}

# Argument parsing
while getopts "p:i:o:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        o ) OUTPUT_DIR="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${OUTPUT_DIR}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
python3 "${PROJECT_PATH}/code/configurate.py" "${INPUT_FILE}" "${OUTPUT_DIR}"
