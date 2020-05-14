#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file -r result_files\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-r Result files paths\n"
    exit 1
}

# Argument parsing
while getopts "p:i:r:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        r ) RESULT_FILES="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${RESULT_FILES}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
mkdir -p "${PROJECT_PATH}/plots"

# POSIX shell scripts do not allow arrays (workaround)
echo "${RESULT_FILES}" | tr ' ' '\n' | while read file; do
    echo "${file}"
    tar -xvf "${file}" -C "${PROJECT_PATH}";
done

python3 "${PROJECT_PATH}/code/plotting.py" "${INPUT_FILE}"

# ls -lR
# cp -R /madminer/plots {plotworkdir}
