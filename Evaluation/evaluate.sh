#!/bin/bash

# Run this from `Evaluation` folder

RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

TEAMS=(Brigadeiro CoreDumpers ExecuteKernel66 GpuComputingOnRustToAnnoyTheProff GPU-faVellas staVOLTA)
# TEAMS=(staVOLTA)

## [[ -f main.zip ]] || wget https://github.com/ThomasPasquali/GPU-computing-hackathon/archive/refs/heads/main.zip

mkdir -p clones
base_dir=$(pwd)

for team in ${TEAMS[@]}; do
    echo -e "${PUR} =========================================  $team  ========================================= ${NC}"
    cd $base_dir
    team_clone_folder="./clones/$team"

    # Clone fresh repo
    # [[ -d $team_clone_folder ]] || unzip main.zip -d $team_clone_folder
    if [[ ! -d $team_clone_folder ]]; then
        mkdir -p $team_clone_folder
        cd clones
        git clone https://github.com/ThomasPasquali/GPU-computing-hackathon.git $team
        cd $team
        git submodule init
        git submodule update distributed_mmio SbatchMan
        cd $base_dir
    fi

    # Substitute team files
    cp -r "./$team/"* "$team_clone_folder"

    cd $team_clone_folder
    make bin/bfs && echo -e "${GRE} Compiled '$team' successfully! ${NC}" || echo -e "${RED}Could NOT compile '$team'! ${NC}"
    ./run_experiments.sh

    echo -e "${PUR} =========================================  $team - DONE!  ========================================= ${NC}\n\n"
done 