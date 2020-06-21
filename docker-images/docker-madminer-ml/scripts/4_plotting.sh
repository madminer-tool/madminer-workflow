#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file -r result_files -o output_dir\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-r Result files paths\n"
    printf "\t-o Workflow output dir\n"
    exit 1
}

# Argument parsing
while getopts "p:i:r:o:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        r ) RESULT_FILES="$OPTARG" ;;
        o ) OUTPUT_DIR="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${RESULT_FILES}" ] || [ -z "${OUTPUT_DIR}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
mkdir -p "${OUTPUT_DIR}/plots"

# POSIX shell scripts do not allow arrays (workaround)
echo "${RESULT_FILES}" | tr ' ' '\n' | while read -r file; do
    echo "${file}"
    tar -xvf "${file}" -C "${OUTPUT_DIR}";
done

python3 "${PROJECT_PATH}/code/plotting.py" "${INPUT_FILE}" "${OUTPUT_DIR}"
