#!/bin/bash


# data_type=(
# "f32"
# "s8"
# "f16"
# "bf16"
# )

ONEDNN_MAX_CPU_ISA=(
"AVX2_VNNI" #256 for int8,f32
"AVX512_CORE_FP16" #512 for int8,fp16,bf16,f32
"DEFAULT"  #AMX1024*8 for bf16 && int8
)

data_type_avx2_vnni=(
"f32"
"s8"
)

data_type_avx512_core_fp16=(
"f32"
"s8"
"f16"
"bf16"
)

data_type_default=(
"s8"
"bf16"
"f32"
)

#循环次数
loop_time=10

ONEDNN_ROOT_PATH="$PWD"

mkdir ${ONEDNN_ROOT_PATH}/log

i=0
for((j=0; j<${#data_type_avx2_vnni[@]}; j++))
do
    echo "ONEDNN_MAX_CPU_ISA:${ONEDNN_MAX_CPU_ISA[i]}"
    echo "data_type:${data_type_avx2_vnni[j]}"
    echo "resnet50"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_avx2_vnni[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_avx2_vnni[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_avx2_vnni[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_avx2_vnni[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done
    
    echo "vgg19"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_avx2_vnni[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_avx2_vnni[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_avx2_vnni[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_avx2_vnni[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done

done
i=1
for((j=0; j<${#data_type_avx512_core_fp16[@]}; j++))
do
    echo "ONEDNN_MAX_CPU_ISA:${ONEDNN_MAX_CPU_ISA[i]}"
    echo "data_type:${data_type_avx512_core_fp16[j]}"
    echo "resnet50"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_avx512_core_fp16[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_avx512_core_fp16[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_avx512_core_fp16[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_avx512_core_fp16[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done

    echo "vgg19"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_avx512_core_fp16[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_avx512_core_fp16[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_avx512_core_fp16[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_avx512_core_fp16[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done
done
i=2
for((j=0; j<${#data_type_default[@]}; j++))
do
    echo "ONEDNN_MAX_CPU_ISA:${ONEDNN_MAX_CPU_ISA[i]}"
    echo "data_type:${data_type_default[j]}"
    
    echo "resnet50"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_default[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_default[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dt=${data_type_default[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_true_resnet_50 > ${ONEDNN_ROOT_PATH}/log/resnet_50_${data_type_default[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done

    echo "vgg19"
    for ((k=0; k<loop_time; k++))
    do
        echo "循环次数：${k}"
        sudo OMP_NUM_THREADS=1 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=2 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_default[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_default[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread1.info
            
        sudo OMP_NUM_THREADS=2 ONEDNN_LOOP=${k}  ONEDNN_MAX_CPU_ISA=${ONEDNN_MAX_CPU_ISA[i]} \
                ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=perf_info \
                ONEDNN_VERBOSE_AFFINITY_CONV_CPUID=0,2 ONEDNN_VERBOSE_AFFINITY_OTHER_CPUID=4,6 ONEDNN_VERBOSE_AFFINITY=yy \
                ${ONEDNN_ROOT_PATH}/build/tests/benchdnn/benchdnn --conv --dir=FWD_B --dt=${data_type_default[j]} --batch=${ONEDNN_ROOT_PATH}/tests/benchdnn/inputs/conv/shapes_vgg_19 > ${ONEDNN_ROOT_PATH}/log/vgg_19_${data_type_default[j]}_${ONEDNN_MAX_CPU_ISA[i]}_loop_${k}_thread2.info
    done
done

