#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -c config_file -i input_file -e events_file\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-c Configuration file path\n"
    printf "\t-i Workflow input file\n"
    printf "\t-e Events file path\n"
    exit 1
}

# Argument parsing
while getopts "p:c:i:e:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        c ) CONFIG_FILE="$OPTARG" ;;
        i ) INPUT_FILE="$OPTARG" ;;
        e ) EVENTS_FILE="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${CONFIG_FILE}" ] || [ -z "${INPUT_FILE}" ] || [ -z "${EVENTS_FILE}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Define auxiliary variables
EXTRACT_PATH="${PROJECT_PATH}/extract"
DATA_PATH="${PROJECT_PATH}/data"


# Perform actions
mkdir -p "${EXTRACT_PATH}"
tar -xvf "${EVENTS_FILE}" -C "${EXTRACT_PATH}"
mv "${EXTRACT_PATH}/madminer/cards/benchmark_"*".dat" "${EXTRACT_PATH}/madminer/cards/benchmark.dat"

mkdir -p "${DATA_PATH}"
python3 "${PROJECT_PATH}/code/delphes.py" \
    "${CONFIG_FILE}" \
    "${EXTRACT_PATH}/Events/run_01" \
    "${INPUT_FILE}" \
    "${EXTRACT_PATH}/madminer/cards/benchmark.dat"

# cp  -R /madminer/data/* {dworkdir}
