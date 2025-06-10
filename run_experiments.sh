#!/bin/bash

# Always run this script from the root of the repo

source env.sh

RUN_SMALL=true
RUN_LARGE=true
RUN_G500=true

# Parse CLI arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --no-small-diam) RUN_SMALL=false ;;
        --no-large-diam) RUN_LARGE=false ;;
        --no-graph500) RUN_G500=false ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Setup environment
if [[ -z $SbM_HOME ]]; then
    echo -e "${GRE}Setting up experiments environment...${NC}"
    if [[ ! -f SbatchMan/sourceFile.sh ]]; then
        echo -e "${GRE}Initializing SLURM configurations...${NC}"
        cd SbatchMan
        ./initEnv.sh
        cd ..
        source SbatchMan/sourceFile.sh
        # 1 node, 1 CPU, 1 GPU, no MPI
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e BFS_smallD -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton  
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e BFS_largeD -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton  
        SbatchMan/newExperiment.sh -p "edu-short" -t 00:04:00 -e BFS_g500   -n 1 -c 1 -g 1 -d 1 -w edu01 -b $BIN # -a hackaton 
    else
        source SbatchMan/sourceFile.sh
    fi
fi

echo -e "${GRE}Building $BIN...${NC}"
make clean $BIN
source SbatchMan/submit.sh
my_hostname=$(${SbM_UTILS}/hostname.sh)

check_and_move_old_exp () {
    if [[ -d "${SbM_HOME}/sout/$my_hostname/$1" ]] && [[ $(find "${SbM_HOME}/sout/$my_hostname/$1" -type f | wc -l) -gt 0 ]]; then
        echo -e "${PUR}Old experiment data found for $1. Do you want to archive it? (Y/n)${NC}"
        read -r response
        if [[ "$response" == "n" || "$response" == "N" ]]; then
            return 1
        else
           SbatchMan/utils/archiveExperiments.sh $1
           return 0
        fi
    fi
}

if $RUN_SMALL; then
    expname=BFS_smallD
    check_and_move_old_exp $expname
    if [[ $? -eq 0 ]]; then
        echo -e "${GRE}%% Running tests on small-diameter graphs %%${NC}"
        for gi in ${!GRAPHS_SMALL_D[@]}; do
            graph=${GRAPHS_SMALL_D[$gi]}
            echo "----- Testing '$(basename "${graph%.*}")' graph -----"
            SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS
            echo "JOB ID: ${job_id}"
        done
    fi
fi

if $RUN_LARGE; then
    expname=BFS_largeD
    check_and_move_old_exp $expname
    if [[ $? -eq 0 ]]; then
        echo -e "${GRE}%% Running tests on large-diameter graphs %%${NC}"
        for gi in ${!GRAPHS_LARGE_D[@]}; do
            graph=${GRAPHS_LARGE_D[$gi]}
            echo "----- Testing '$(basename "${graph%.*}")' graph -----"
            SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS
            echo "JOB ID: ${job_id}"
        done
    fi
fi

if $RUN_G500; then
    expname=BFS_g500
    check_and_move_old_exp $expname
    if [[ $? -eq 0 ]]; then
        echo -e "${GRE}%% Running tests on Graph500 graphs %%${NC}"
        for gi in ${!GRAPHS_G500[@]}; do
            graph=${GRAPHS_G500[$gi]}
            echo "----- Testing '$(basename "${graph%.*}")' graph -----"
            SbM_submit_function --verbose --expname $expname --binary $BIN -f "$MTX_PATH/$graph" -n $ITERATIONS
            echo "JOB ID: ${job_id}"
        done
    fi
fi
