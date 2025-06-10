## !! PLEASE MODIFY THE VALUE OF "GROUP_NAME"
## Write here the name of your group
## Please use the name you submitted in the registration form  
export GROUP_NAME="staVOLTA"
## !!

RED='\033[0;31m'
PUR='\033[0;35m'
GRE='\033[0;32m'
NC='\033[0m' # No Color

export BIN=bin/bfs
export ITERATIONS=15
export HOST="baldo"
export SHARED_DIR="/data/hackathon"
export MTX_PATH="${SHARED_DIR}/datasets"

# Read graph paths from matrices_list.txt in each subfolder
GRAPHS_SMALL_D=()
while IFS= read -r line; do
    GRAPHS_SMALL_D+=("$line")
done < "$MTX_PATH/small_diameter/matrices_list.txt"

GRAPHS_LARGE_D=()
while IFS= read -r line; do
    GRAPHS_LARGE_D+=("$line")
done < "$MTX_PATH/large_diameter/matrices_list.txt"

GRAPHS_G500=()
while IFS= read -r line; do
    GRAPHS_G500+=("$line")
done < "$MTX_PATH/graph500/matrices_list.txt"

export GRAPHS_SMALL_D
export GRAPHS_LARGE_D
export GRAPHS_G500
