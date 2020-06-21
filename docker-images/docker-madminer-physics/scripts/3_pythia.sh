#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -m madgraph_folder -s signal_folder -l logs_folder -z zip_file -o output_dir\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-m MadGraph folder sub-path\n"
    printf "\t-s Signal folder sub-path\n"
    printf "\t-l Logs folder sub-path\n"
    printf "\t-z Zip file path\n"
    printf "\t-o Workflow output dir\n"
    exit 1
}

# Argument parsing
while getopts "p:m:s:l:z:o:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        m ) MADGRAPH_FOLDER="$OPTARG" ;;
        s ) SIGNAL_FOLDER="$OPTARG" ;;
        l ) LOGS_FOLDER="$OPTARG" ;;
        z ) ZIP_FILE="$OPTARG" ;;
        o ) OUTPUT_DIR="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${MADGRAPH_FOLDER}" ] || [ -z "${SIGNAL_FOLDER}" ] || \
    [ -z "${LOGS_FOLDER}" ] || [ -z "${ZIP_FILE}" ] || [ -z "${OUTPUT_DIR}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Define auxiliary variables
MADGRAPH_ABS_PATH="${PROJECT_PATH}/${MADGRAPH_FOLDER}"
SIGNAL_ABS_PATH="${OUTPUT_DIR}/${SIGNAL_FOLDER}"
LOGS_ABS_PATH="${OUTPUT_DIR}/${LOGS_FOLDER}"


# Cleanup previous files (useful when run locally)
rm -rf "${OUTPUT_DIR}/events"
rm -rf "${SIGNAL_ABS_PATH}/Events"
rm -rf "${SIGNAL_ABS_PATH}/madminer"
rm -rf "${SIGNAL_ABS_PATH}/rw_me"

mkdir -p "${OUTPUT_DIR}/events"
mkdir -p "${SIGNAL_ABS_PATH}/Events"
mkdir -p "${SIGNAL_ABS_PATH}/madminer"
mkdir -p "${SIGNAL_ABS_PATH}/rw_me"


# Perform actions
tar -xvf "${ZIP_FILE}" -C "${SIGNAL_ABS_PATH}/madminer"

mkdir -p "${LOGS_ABS_PATH}"
sh "${SIGNAL_ABS_PATH}/madminer/scripts/run"*".sh" "${MADGRAPH_ABS_PATH}" "${SIGNAL_ABS_PATH}" "${LOGS_ABS_PATH}"

tar -czvf "${OUTPUT_DIR}/events/Events.tar.gz" \
    -C "${SIGNAL_ABS_PATH}" \
    "Events" \
    "madminer/cards"
