#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -i [input_file_1, input_file_2, ...]\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-i Input files coming from Delphes\n"
    exit 1
}

# Argument parsing
while getopts "p:i:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        i ) INPUT_FILES="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${INPUT_FILES}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Perform actions
echo "${INPUT_FILES}"
python3 "${PROJECT_PATH}/code/combine.py" "${INPUT_FILES}"

# cp -R /madminer/combined_delphes.h5 {cworkdir}
