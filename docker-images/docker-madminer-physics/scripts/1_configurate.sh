#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i input_file\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Workflow input file\n"
    exit 1
}

# Argument parsing
while getopts "p:i:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILE}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
python3 "${PROJECT_PATH}/code/configurate.py" "${INPUT_FILE}"

# cp -R /madminer/data/*.h5 {workdir}
