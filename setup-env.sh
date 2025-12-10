
# Jericho Setup: https://jericho-py.readthedocs.io/en/latest/tutorial_quick.html

# Acquire games
if [ ! -d "z-machine-games-master" ];
then
    wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip
    unzip master.zip
    rm master.zip
fi

export LD_LIBRARY_PATH='./env/lib/python3.13/site-packages/jericho'