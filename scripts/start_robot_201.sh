#!/usr/bin/env bash -l

if [[ "$CONDA_DEFAULT_ENV" != "polymetis" ]]
then
    echo Activating polymetis env

    eval "$(conda shell.bash hook)"
    conda activate polymetis

    if [[ "$CONDA_DEFAULT_ENV" != "polymetis" ]]
    then
        echo Failed to activate conda environment.
        exit 1
    fi
fi

launch_robot.py ip=10.10.10.210 robot_model=franka_panda robot_client=franka_hardware robot_client.executable_cfg.robot_ip=10.10.10.201
