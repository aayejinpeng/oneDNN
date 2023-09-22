#!/bin/bash


data_type=(
"f32"
"f64"
"f32f32s8"
"u8s8f32"
"u8s8s32"
"u8s8s8"
"u8s8u8"
"s8s8f32"
"s8s8s32"
"s8s8s8"
"s8s8u8"
"f32_wino"
"f16"
"f16f16s8"
"bf16bf16bf16"
"bf16bf16f32"
"bf16f32bf16"
"f32bf16bf16"
)

#循环次数
loop_time=1

ONEDNN_ROOT_PATH="$PWD"

mkdir ${ONEDNN_ROOT_PATH}/log

for ((i=0; i<${#data_type[@]}; i++))
do
    echo "数据类型：${data_type[i]}"
    for ((j=0; j<loop_time; j++))
    do
        echo "循环次数：${j}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${j}  \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type[i]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type[i]}_loop_${j}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${j}  \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type[i]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type[i]}_loop_${j}_thread2.info
    done

    # for ((j=0; j<loop_time; j++))
    # do
    #     echo "循环次数：${j}"
    #     sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${j}  \
    #             ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
    #             ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
    #             ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type[i]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type[i]}_loop_${j}_thread1.info
            
        
    #     sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${j}  \
    #             ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
    #             ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
    #             ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type[i]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_resnet_50 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type[i]}_loop_${j}_thread2.info
        
    # done
done
