#!/usr/bin/env sh


# Define help function
helpFunction()
{
    printf "\n"
    printf "Usage: %s -p project_path -s signal_folder -j num_jobs -c config_file\n" "${0}"
    printf "\t-p Project top-level path\n"
    printf "\t-s Signal folder sub-path\n"
    printf "\t-j Number of jobs\n"
    printf "\t-c Configuration file path\n"
    exit 1
}

# Argument parsing
while getopts "p:s:j:c:" opt
do
    case "$opt" in
        p ) PROJECT_PATH="$OPTARG" ;;
        s ) SIGNAL_FOLDER="$OPTARG" ;;
        j ) NUM_JOBS="$OPTARG" ;;
        c ) CONFIG_FILE="$OPTARG" ;;
        ? ) helpFunction ;;
    esac
done

if [ -z "${PROJECT_PATH}" ] || [ -z "${SIGNAL_FOLDER}" ] || [ -z "${NUM_JOBS}" ] || [ -z "${CONFIG_FILE}" ]
then
    echo "Some or all of the parameters are empty";
    helpFunction
fi


# Define auxiliary variables
SIGNAL_ABS_PATH="${PROJECT_PATH}/${SIGNAL_FOLDER}"


# Perform actions
python3 "${PROJECT_PATH}/code/generate.py" "${NUM_JOBS}" "${CONFIG_FILE}"

for i in $(seq 0 $((NUM_JOBS-1))); do
    tar -czvf "${PROJECT_PATH}/folder_${i}.tar.gz" \
        -C "${SIGNAL_ABS_PATH}/madminer" \
        "scripts/run_${i}.sh" \
        "cards/benchmark_${i}.dat" \
        "cards/mg_commands_${i}.dat" \
        "cards/param_card_${i}.dat" \
        "cards/pythia8_card_${i}.dat" \
        "cards/reweight_card_${i}.dat" \
        "cards/run_card_${i}.dat"
done

# cp -R /madminer/folder_*.tar.gz {workdir}
