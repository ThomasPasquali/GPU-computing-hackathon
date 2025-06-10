#!/bin/bash

# Run this from `Evaluation` folder

RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

TEAMS=(Brigadeiro CoreDumpers ExecuteKernel66 GpuComputingOnRustToAnnoyTheProff GPU-faVellas staVOLTA)
# TEAMS=(staVOLTA)

base_dir=$(pwd)

for team in ${TEAMS[@]}; do
    echo -e "${PUR} =========================================  $team  ========================================= ${NC}"
    cd $base_dir
    team_clone_folder="./clones/$team"
    if [[ ! -d $team_clone_folder ]]; then
        echo -e "${RED}NO TEAM FOLDER${NC}"
        continue
    fi
    cd $team_clone_folder
    ./submit_results.sh
    echo -e "${PUR} =========================================  $team - DONE!  ========================================= ${NC}\n\n"
done 