/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <string>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <unistd.h>

#include "c_types_map.hpp"
#include "engine.hpp"

#if defined(DNNL_ENABLE_ITT_TASKS)
#include "ittnotify.hpp"
#endif

#include "primitive.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_exec_types.hpp"
#include "primitive_iface.hpp"
#include "profiler.hpp"
#include "reorder_pd.hpp"
#include "scratchpad_debug.hpp"
#include "stack_checker.hpp"
#include "stream.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::status;
using namespace dnnl::impl::primitive_kind;

namespace {
// XXX: this is a huge hammer. This disables all and any msan checks on
// primitives outputs.
//
// A proper approach would be an implementation-specific unpoisoning.
void unpoison_outputs(const exec_args_t &args) {
    for (const auto &arg : args) {
        if (arg.second.is_const) continue;
        auto *mem = arg.second.mem;
        void *p;
        mem->get_data_handle(&p);
        size_t s = memory_desc_wrapper(*mem->md()).size();
        msan_unpoison(p, s);
    }
}
} // namespace

namespace dnnl {
namespace impl {

status_t primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface,
        const cache_blob_t &cache_blob = cache_blob_t()) {

    std::pair<primitive_iface_t *, bool> p_iface;

    if (verbose_has_create_profile()) {
        double start_ms = get_msec();
        CHECK(primitive_desc_iface->create_primitive_iface(
                p_iface, cache_blob));
        double duration_ms = get_msec() - start_ms;

        const char *str = p_iface.second ? ":cache_hit" : ":cache_miss";
        if (cache_blob) str = ":from_cache_blob";

        VPROF(start_ms, create, str, p_iface.first->pd()->info(), duration_ms);
    } else {
        CHECK(primitive_desc_iface->create_primitive_iface(
                p_iface, cache_blob));
    }
    return safe_ptr_assign((*primitive_iface), p_iface.first);
}

status_t primitive_execute(
        const primitive_iface_t *primitive_iface, exec_ctx_t &ctx) {
    auto stream = ctx.stream();
    status_t status = success;

#if defined(DNNL_ENABLE_ITT_TASKS)
    const bool enable_itt = itt::get_itt(itt::__itt_task_level_low);
    if (enable_itt)
        itt::primitive_task_start(primitive_iface->pd()->impl()->kind());
#endif

    if (verbose_has_exec_profile()) {
        stream->wait();
		//todo: add cpu_affinity
		static std::string verbose_affinity_env = getenv_string_user("VERBOSE_AFFINITY");
		// printf("verbose_affinity_env is %s\n", verbose_affinity_env.c_str());
		//conv(x)_other(y)
		if (verbose_affinity_env == "y")
		{
			static std::string conv_cpuid_env = getenv_string_user("VERBOSE_AFFINITY_CONV_CPUID");
			static std::string other_cpuid_env = getenv_string_user("VERBOSE_AFFINITY_OTHER_CPUID");
			int conv_cpuid = std::stoi(conv_cpuid_env);
			int other_cpuid = std::stoi(other_cpuid_env);
			if (primitive_iface->pd()->impl()->kind() == primitive_kind::convolution)
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
				CPU_SET(conv_cpuid, &mask);
				int cpuid = sched_getcpu();
				sched_setaffinity(0, sizeof(mask), &mask);
				while(1)
				{
					int tmp_cpuid = sched_getcpu();
					if (tmp_cpuid == conv_cpuid)
					{
						// printf("conv_primitive cpuid is changed from %d to %d\n", cpuid, conv_cpuid);
						break;
					}
				}				
			}
			else
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
				CPU_SET(other_cpuid, &mask);
				int cpuid = sched_getcpu();
				sched_setaffinity(0, sizeof(mask), &mask);
				while(1)
				{
					int tmp_cpuid = sched_getcpu();
					if (tmp_cpuid == other_cpuid)
					{
						// printf("other_primitive cpuid is changed from %d to %d\n", cpuid, other_cpuid);
						break;
					}
				}
			}
		}else if (verbose_affinity_env == "yy")
		{
			static std::string conv_cpuid_env = getenv_string_user("VERBOSE_AFFINITY_CONV_CPUID");
			static std::string other_cpuid_env = getenv_string_user("VERBOSE_AFFINITY_OTHER_CPUID");
		    std::vector<int> conv_cpuid;
			std::vector<int> other_cpuid;
            std::istringstream ss(conv_cpuid_env);
            std::string token;
            while (std::getline(ss, token, ',')) {
                conv_cpuid.push_back(std::stoi(token));
            }
            std::istringstream ss2(other_cpuid_env);
            std::string token2;
            while (std::getline(ss2, token2, ',')) {
                other_cpuid.push_back(std::stoi(token2));
            }
            
			if (primitive_iface->pd()->impl()->kind() == primitive_kind::convolution)
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
                for (int i = 0; i < conv_cpuid.size(); i++)
                {
                    CPU_SET(conv_cpuid[i], &mask);
                }
				int cpuid = sched_getcpu();
				sched_setaffinity(0, sizeof(mask), &mask);
				while(1)
				{
					int tmp_cpuid = sched_getcpu();
					if (CPU_ISSET(tmp_cpuid, &mask))
                    {
                        // printf("conv_primitive cpuid is changed from %d to %d\n", cpuid, tmp_cpuid);
                        break;
                    }
				}
			}
			else
			{
				cpu_set_t mask;
				CPU_ZERO(&mask);
                for (int i = 0; i < other_cpuid.size(); i++)
                {
                    CPU_SET(other_cpuid[i], &mask);
                }
				int cpuid = sched_getcpu();
				sched_setaffinity(0, sizeof(mask), &mask);
				while(1)
				{
					int tmp_cpuid = sched_getcpu();
					if (CPU_ISSET(tmp_cpuid, &mask))
                    {
                        // printf("other_primitive cpuid is changed from %d to %d\n", cpuid, tmp_cpuid);
                        break;
                    }
				}
			}
		}
        
        //todo: add cache perf
		static std::string verbose_more_env = getenv_string_user("VERBOSE_MORE");
                if (verbose_more_env == "perf_info")
        {
            static std::string perf_info_loop = getenv_string_user("LOOP");
            struct perf_event_attr pe_l1_cache_write_acc;
		    struct perf_event_attr pe_l1_cache_read_acc;
            struct perf_event_attr pe_l1_cache_read_miss;
            struct perf_event_attr pe_CPU_CYCLES;

            memset(&pe_CPU_CYCLES, 0, sizeof(struct perf_event_attr));
            pe_CPU_CYCLES.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_CPU_CYCLES.size = sizeof(struct perf_event_attr);
            pe_CPU_CYCLES.config = PERF_COUNT_HW_CPU_CYCLES;
            pe_CPU_CYCLES.exclude_kernel = 1;
            pe_CPU_CYCLES.pinned = 1;
            pe_CPU_CYCLES.exclude_idle = 1;
            pe_CPU_CYCLES.exclude_hv = 1;
            // pe_CPU_CYCLES.inherit = 1;
            // pe_CPU_CYCLES.inherit_stat = 1;
            //write read miss time
            //52622,3971893,51064,3837357 1core
            //31302,2217995,41871,2165214 2core
            //52622,3971891,44767,3697911 1core 


            memset(&pe_l1_cache_write_acc, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_write_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_write_acc.size = sizeof(struct perf_event_attr);
            pe_l1_cache_write_acc.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_l1_cache_write_acc.exclude_kernel = 1;
            pe_l1_cache_write_acc.pinned = 1;
            pe_l1_cache_write_acc.exclude_idle = 1;
            pe_l1_cache_write_acc.exclude_hv = 1;
            // pe_l1_cache_write_acc.inherit = 1;
            // pe_l1_cache_write_acc.inherit_stat = 1;

            memset(&pe_l1_cache_read_miss, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_read_miss.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_read_miss.size = sizeof(struct perf_event_attr);
            pe_l1_cache_read_miss.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            pe_l1_cache_read_miss.exclude_kernel = 1;
            pe_l1_cache_read_miss.pinned = 1;
            pe_l1_cache_read_miss.exclude_idle = 1;
            pe_l1_cache_read_miss.exclude_hv = 1;

            memset(&pe_l1_cache_read_acc, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_read_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_read_acc.size = sizeof(struct perf_event_attr);
            pe_l1_cache_read_acc.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_l1_cache_read_acc.exclude_kernel = 1;
            pe_l1_cache_read_acc.pinned = 1;
            pe_l1_cache_read_acc.exclude_idle = 1;
            pe_l1_cache_read_acc.exclude_hv = 1;
            pid_t pid = getpid();
            int fd_l1_w_acc = syscall(__NR_perf_event_open, &pe_l1_cache_write_acc, pid, -1, -1, 0);
            int fd_l1_r_miss = syscall(__NR_perf_event_open, &pe_l1_cache_read_miss, pid, -1, -1, 0);
            int fd_l1_r_acc = syscall(__NR_perf_event_open, &pe_l1_cache_read_acc, pid, -1, -1, 0);
            int fd_cpu_cycle = syscall(__NR_perf_event_open, &pe_CPU_CYCLES, pid, -1, -1, 0);

            double start_ms = get_msec();
            status = stream->enqueue_primitive(primitive_iface, ctx);
            stream->wait();
            double duration_ms = get_msec() - start_ms;
            unsigned long count[20];
            read(fd_l1_w_acc, &count[7], sizeof(unsigned long));
            read(fd_l1_r_miss, &count[8], sizeof(unsigned long));
            read(fd_l1_r_acc, &count[9], sizeof(unsigned long));

            read(fd_cpu_cycle, &count[15], sizeof(unsigned long));

            // printf("L1 hw_l1cache write acc: %'lu\n", count[7]);
            // printf("L1 hw_l1cache read misses: %'lu\n", count[8]);
            // printf("L1 hw_l1cache read acc: %'lu\n", count[9]);

            // printf("cpu_cycle: %'lu\n", count[15]);

            // printf("L1 hw_l1cache read/write ratio: r%lf:w%lf\n", (count[9]+count[7]==0)?(1.0):(1.0*count[9])/(count[9]+count[7]),(1.0*count[7])/(count[9]+count[7]));
            // printf("L1 hw_l1cache read_miss ratio: %lf\n", (count[9]==0)?(1.0):((1.0*count[8])/count[9]));
            // printf("L1 hw_l1cache read_miss(write alloc) ratio: %lf\n", (count[9]==0)?(1.0):((1.0*count[8])/(count[9]+count[7])));

            close(fd_l1_w_acc);
            close(fd_l1_r_miss);
            close(fd_l1_r_acc);
            close(fd_cpu_cycle);
            unsigned long l1cache_write_acc_time = count[7];
            unsigned long l1cache_read_miss_time = count[8];
            unsigned long l1cache_read_acc_time = count[9];
            unsigned long cpu_cycle_time = count[15];

            // VPROF(start_ms, exec, VERBOSE_profile, primitive_iface->pd()->info(),
            //         duration_ms);

            // VPROF_YJP(start_ms, exec, VERBOSE_profile, primitive_iface->pd()->info(),
            //         duration_ms, l1cache_write_acc_time, l1cache_read_miss_time, l1cache_read_acc_time, cpu_cycle_time);
            std::vector<std::string> result;
            std::istringstream ss(primitive_iface->pd()->info());
            std::string token;

            while (std::getline(ss, token, ',')) {
                result.push_back(token);
                // std::cout << token << std::endl;
            }

            printf("%s,%s,%s,%s,%s,%g,%lu,%lu,%lu,%lu\n",perf_info_loop.c_str(),result[1].c_str(),result[2].c_str(),result[result.size()-2].c_str(),result[result.size()-1].c_str(),duration_ms,l1cache_write_acc_time,l1cache_read_acc_time,l1cache_read_miss_time,cpu_cycle_time);
            // printf("%s,",primitive_iface->pd()->info());
            fflush(stdout);
            
        }
        else
        {
            int verbose_more_setting = 0;
            enum more_info_settings : int {
                No_info = 0,
                LLC_W_Info = 1,
                LLC_R_Info = 2,
                L1D_Info = 10,
                Cycles_Info = 100,
                Cycles_stall_Info = 101
            };
            if (verbose_more_env == "llc_w_info")
            {
                verbose_more_setting = more_info_settings::LLC_W_Info;
            }
            else if (verbose_more_env == "llc_r_info")
            {
                verbose_more_setting = more_info_settings::LLC_R_Info;
            }
            else if (verbose_more_env == "l1d_info")
            {
                verbose_more_setting = more_info_settings::L1D_Info;
            }
            else if (verbose_more_env == "cycles_info")
            {
                verbose_more_setting = more_info_settings::Cycles_Info;
            }
            else if (verbose_more_env == "cycles_stall_info")
            {
                verbose_more_setting = more_info_settings::Cycles_stall_Info;
            }
            else
            {
                // printf("verbose_more_env.length = %d\nverbose_more_env = %s\n",verbose_more_env.length(),verbose_more_env.c_str());
                verbose_more_setting = more_info_settings::No_info;
            }

            struct perf_event_attr pe_llc_cache_write_acc;
            struct perf_event_attr pe_llc_cache_write_miss;
            struct perf_event_attr pe_llc_cache_read_acc;
            struct perf_event_attr pe_llc_cache_read_miss;
            struct perf_event_attr pe_l1_cache_write_acc;
            struct perf_event_attr pe_l1_cache_read_acc;
            struct perf_event_attr pe_l1_cache_read_miss;
            struct perf_event_attr pe_CPU_REF_CYCLES;
            struct perf_event_attr pe_CPU_CYCLES;
            struct perf_event_attr pe_BUS_CYCLES;
            struct perf_event_attr pe_STALLED_CYCLES_FRONTEND;
            struct perf_event_attr pe_STALLED_CYCLES_BACKEND;

            memset(&pe_STALLED_CYCLES_FRONTEND, 0, sizeof(struct perf_event_attr));
            pe_STALLED_CYCLES_FRONTEND.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_STALLED_CYCLES_FRONTEND.size = sizeof(struct perf_event_attr);
            pe_STALLED_CYCLES_FRONTEND.config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
            pe_STALLED_CYCLES_FRONTEND.exclude_kernel = 1;
            pe_STALLED_CYCLES_FRONTEND.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_STALLED_CYCLES_BACKEND, 0, sizeof(struct perf_event_attr));
            pe_STALLED_CYCLES_BACKEND.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_STALLED_CYCLES_BACKEND.size = sizeof(struct perf_event_attr);
            pe_STALLED_CYCLES_BACKEND.config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
            pe_STALLED_CYCLES_BACKEND.exclude_kernel = 1;
            pe_STALLED_CYCLES_BACKEND.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_CPU_REF_CYCLES, 0, sizeof(struct perf_event_attr));
            pe_CPU_REF_CYCLES.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_CPU_REF_CYCLES.size = sizeof(struct perf_event_attr);
            pe_CPU_REF_CYCLES.config = PERF_COUNT_HW_REF_CPU_CYCLES;
            pe_CPU_REF_CYCLES.exclude_kernel = 1;
            pe_CPU_REF_CYCLES.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_CPU_CYCLES, 0, sizeof(struct perf_event_attr));
            pe_CPU_CYCLES.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_CPU_CYCLES.size = sizeof(struct perf_event_attr);
            pe_CPU_CYCLES.config = PERF_COUNT_HW_CPU_CYCLES;
            pe_CPU_CYCLES.exclude_kernel = 1;
            pe_CPU_CYCLES.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_BUS_CYCLES, 0, sizeof(struct perf_event_attr));
            pe_BUS_CYCLES.type = PERF_TYPE_HARDWARE; // 使用硬件缓存事件类型
            pe_BUS_CYCLES.size = sizeof(struct perf_event_attr);
            pe_BUS_CYCLES.config = PERF_COUNT_HW_BUS_CYCLES;
            pe_BUS_CYCLES.exclude_kernel = 1;
            pe_BUS_CYCLES.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_llc_cache_write_miss, 0, sizeof(struct perf_event_attr));
            pe_llc_cache_write_miss.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_llc_cache_write_miss.size = sizeof(struct perf_event_attr);
            pe_llc_cache_write_miss.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_llc_cache_write_miss.exclude_kernel = 1;
            pe_llc_cache_write_miss.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_llc_cache_write_acc, 0, sizeof(struct perf_event_attr));
            pe_llc_cache_write_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_llc_cache_write_acc.size = sizeof(struct perf_event_attr);
            pe_llc_cache_write_acc.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_llc_cache_write_acc.exclude_kernel = 1;
            pe_llc_cache_write_acc.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_llc_cache_read_miss, 0, sizeof(struct perf_event_attr));
            pe_llc_cache_read_miss.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_llc_cache_read_miss.size = sizeof(struct perf_event_attr);
            pe_llc_cache_read_miss.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            pe_llc_cache_read_miss.exclude_kernel = 1;
            pe_llc_cache_read_miss.pinned = 1;
            // pe_llc_write_miss.exclude_idle = 1;
            // pe_llc_write_miss.exclude_hv = 1;

            memset(&pe_llc_cache_read_acc, 0, sizeof(struct perf_event_attr));
            pe_llc_cache_read_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_llc_cache_read_acc.size = sizeof(struct perf_event_attr);
            pe_llc_cache_read_acc.config = (PERF_COUNT_HW_CACHE_LL) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_llc_cache_read_acc.exclude_kernel = 1;
            pe_llc_cache_read_acc.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_l1_cache_write_acc, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_write_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_write_acc.size = sizeof(struct perf_event_attr);
            pe_l1_cache_write_acc.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_l1_cache_write_acc.exclude_kernel = 1;
            pe_l1_cache_write_acc.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;

            memset(&pe_l1_cache_read_miss, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_read_miss.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_read_miss.size = sizeof(struct perf_event_attr);
            pe_l1_cache_read_miss.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            pe_l1_cache_read_miss.exclude_kernel = 1;
            pe_l1_cache_read_miss.pinned = 1;
            // pe_llc_write_miss.exclude_idle = 1;
            // pe_llc_write_miss.exclude_hv = 1;

            memset(&pe_l1_cache_read_acc, 0, sizeof(struct perf_event_attr));
            pe_l1_cache_read_acc.type = PERF_TYPE_HW_CACHE; // 使用硬件缓存事件类型
            pe_l1_cache_read_acc.size = sizeof(struct perf_event_attr);
            pe_l1_cache_read_acc.config = (PERF_COUNT_HW_CACHE_L1D) | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            pe_l1_cache_read_acc.exclude_kernel = 1;
            pe_l1_cache_read_acc.pinned = 1;
            // pe_llc_write_acc.exclude_idle = 1;
            // pe_llc_write_acc.exclude_hv = 1;
            pid_t pid = getpid();
            int fd_llc_w_miss = (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):(syscall(__NR_perf_event_open, &pe_llc_cache_write_miss, pid, -1, -1, 0));
            int fd_llc_w_acc = (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):(syscall(__NR_perf_event_open, &pe_llc_cache_write_acc, pid, -1, -1, 0));
            int fd_llc_r_miss = (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):(syscall(__NR_perf_event_open, &pe_llc_cache_read_miss, pid, -1, -1, 0));
            int fd_llc_r_acc = (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):(syscall(__NR_perf_event_open, &pe_llc_cache_read_acc, pid, -1, -1, 0));
            int fd_l1_w_acc = (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):(syscall(__NR_perf_event_open, &pe_l1_cache_write_acc, pid, -1, -1, 0));
            int fd_l1_r_miss = (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):(syscall(__NR_perf_event_open, &pe_l1_cache_read_miss, pid, -1, -1, 0));
            int fd_l1_r_acc = (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):(syscall(__NR_perf_event_open, &pe_l1_cache_read_acc, pid, -1, -1, 0));
            int fd_cpu_ref_cycle = (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):(syscall(__NR_perf_event_open, &pe_CPU_REF_CYCLES, pid, -1, -1, 0));
            int fd_cpu_cycle = (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):(syscall(__NR_perf_event_open, &pe_CPU_CYCLES, pid, -1, -1, 0));
            int fd_bus_cycle = (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):(syscall(__NR_perf_event_open, &pe_BUS_CYCLES, pid, -1, -1, 0));
            int fd_stall_frontend_cycle = (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):(syscall(__NR_perf_event_open, &pe_STALLED_CYCLES_FRONTEND, pid, -1, -1, 0));
            int fd_stall_backtend_cycle = (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):(syscall(__NR_perf_event_open, &pe_STALLED_CYCLES_BACKEND, pid, -1, -1, 0));

            double start_ms = get_msec();
            status = stream->enqueue_primitive(primitive_iface, ctx);
            stream->wait();
            double duration_ms = get_msec() - start_ms;
            unsigned long count[20];
            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):read(fd_llc_w_miss, &count[2], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):read(fd_llc_w_acc, &count[3], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):read(fd_llc_r_miss, &count[4], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):read(fd_llc_r_acc, &count[5], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):read(fd_l1_w_acc, &count[7], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):read(fd_l1_r_miss, &count[8], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):read(fd_l1_r_acc, &count[9], sizeof(unsigned long));

            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):read(fd_cpu_ref_cycle, &count[14], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):read(fd_cpu_cycle, &count[15], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):read(fd_bus_cycle, &count[17], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):read(fd_stall_frontend_cycle, &count[18], sizeof(unsigned long));
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):read(fd_stall_backtend_cycle, &count[19], sizeof(unsigned long));

            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):printf("LLC hw_cache write misses: %'lu\n", count[2]);
            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):printf("LLC hw_cache write acc: %'lu\n", count[3]);
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):printf("LLC hw_cache read misses: %'lu\n", count[4]);
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):printf("LLC hw_cache read acc: %'lu\n", count[5]);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache write acc: %'lu\n", count[7]);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache read misses: %'lu\n", count[8]);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache read acc: %'lu\n", count[9]);

            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):printf("cpu_ref_cycle: %'lu\n", count[14]);
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):printf("cpu_cycle: %'lu\n", count[15]);
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):printf("bus_cycle: %'lu\n", count[17]);
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):printf("stall_frontend_cycle: %'lu\n", count[18]);
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):printf("stall_backtend_cycle: %'lu\n", count[19]);

            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):printf("LLC hw_cache write_miss ratio: %lf\n", ((count[3]==0)?(1.0):(1.0*count[2])/count[3]));
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):printf("LLC hw_cache read_miss ratio: %lf\n", (count[5]==0)?(1.0):((1.0*count[4])/count[5]));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache read/write ratio: r%lf:w%lf\n", (count[9]+count[7]==0)?(1.0):(1.0*count[9])/(count[9]+count[7]),(1.0*count[7])/(count[9]+count[7]));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache read_miss ratio: %lf\n", (count[9]==0)?(1.0):((1.0*count[8])/count[9]));
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):printf("L1 hw_l1cache read_miss(write alloc) ratio: %lf\n", (count[9]==0)?(1.0):((1.0*count[8])/(count[9]+count[7])));

            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):close(fd_llc_w_miss);
            (verbose_more_setting!=more_info_settings::LLC_W_Info)?(-1):close(fd_llc_w_acc);
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):close(fd_llc_r_miss);
            (verbose_more_setting!=more_info_settings::LLC_R_Info)?(-1):close(fd_llc_r_acc);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):close(fd_l1_w_acc);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):close(fd_l1_r_miss);
            (verbose_more_setting!=more_info_settings::L1D_Info)?(-1):close(fd_l1_r_acc);

            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):close(fd_bus_cycle);
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):close(fd_cpu_cycle);
            (verbose_more_setting!=more_info_settings::Cycles_Info)?(-1):close(fd_cpu_ref_cycle);
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):close(fd_stall_frontend_cycle);
            (verbose_more_setting!=more_info_settings::Cycles_stall_Info)?(-1):close(fd_stall_backtend_cycle);
            
            VPROF(start_ms, exec, VERBOSE_profile, primitive_iface->pd()->info(),
                    duration_ms);
        }
    } else {
        status = stream->enqueue_primitive(primitive_iface, ctx);
    }

#if defined(DNNL_ENABLE_ITT_TASKS)
    if (enable_itt) itt::primitive_task_end();
#endif

    if (msan_enabled) unpoison_outputs(ctx.args());

    return status;
}

} // namespace impl
} // namespace dnnl

// API
status_t dnnl_primitive_create(primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;

#ifdef DNNL_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("dnnl_primitive_create");
    bool is_wino = std::string(primitive_desc_iface->info()).find("wino")
            != std::string::npos;

    if (!is_wino) {
        const cache_blob_t dummy;
        return sc.check(dnnl::impl::primitive_create, primitive_iface,
                primitive_desc_iface, std::ref(dummy));
    }
#endif
    return dnnl::impl::primitive_create(primitive_iface, primitive_desc_iface);
}

status_t dnnl_primitive_create_from_cache_blob(
        primitive_iface_t **primitive_iface,
        const primitive_desc_iface_t *primitive_desc_iface, size_t size,
        const uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, primitive_desc_iface, cache_blob)
            || size == 0) {
        return invalid_arguments;
    }
    const auto ekind = primitive_desc_iface->engine()->kind();
    const auto runtime_kind = primitive_desc_iface->engine()->runtime_kind();
    if (ekind != engine_kind::gpu
            || (ekind == engine_kind::gpu
                    && runtime_kind != runtime_kind::ocl)) {
        return status::unimplemented;
    }

    cache_blob_t cb(const_cast<uint8_t *>(cache_blob), size);
    return dnnl::impl::primitive_create(
            primitive_iface, primitive_desc_iface, cb);
}

status_t dnnl_primitive_execute(const primitive_iface_t *primitive_iface,
        stream_t *stream, int nargs, const dnnl_exec_arg_t *c_args) {
    bool ok = true && !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && IMPLICATION(nargs > 0, c_args != nullptr);
    if (!ok) return invalid_arguments;

    exec_args_t args;
    status_t status = cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, c_args, args);
    if (status != status::success) return status;

    stream->before_exec_hook();

    exec_ctx_t ctx(stream, std::move(args));
#ifdef DNNL_ENABLE_STACK_CHECKER
    stack_checker::stack_checker_t sc("dnnl_primitive_execute");
    const auto *pd_iface = primitive_iface->pd();
    bool is_wino
            = std::string(pd_iface->info()).find("wino") != std::string::npos;
    if (!is_wino) {
        status = sc.check(
                dnnl::impl::primitive_execute, primitive_iface, std::ref(ctx));
    }
#else
    status = dnnl::impl::primitive_execute(primitive_iface, ctx);
#endif
    stream->after_exec_hook();

    return status;
}

status_t dnnl_primitive_get_primitive_desc(
        const primitive_iface_t *primitive_iface,
        const primitive_desc_iface_t **primitive_desc_iface) {
    if (utils::any_null(primitive_iface, primitive_desc_iface))
        return invalid_arguments;
    return safe_ptr_assign(*primitive_desc_iface, primitive_iface->pd());
}

status_t dnnl_primitive_get_cache_blob(const primitive_iface_t *primitive_iface,
        size_t *size, uint8_t *cache_blob) {
    if (utils::any_null(primitive_iface, size)) {
        return status::invalid_arguments;
    }

    const auto ekind = primitive_iface->engine()->kind();
    const auto runtime_kind = primitive_iface->engine()->runtime_kind();
    if (ekind != engine_kind::gpu
            || (ekind == engine_kind::gpu
                    && runtime_kind != runtime_kind::ocl)) {
        return status::unimplemented;
    }

    if (!cache_blob) {
        size_t sz = 0;
        CHECK(primitive_iface->get_cache_blob_size(&sz));
        (*size) = sz;
        return status::success;
    }

    cache_blob_t cb(cache_blob, *size);
    return primitive_iface->get_cache_blob(cb);
}

status_t dnnl_primitive_destroy(primitive_iface_t *primitive_iface) {
    if (primitive_iface != nullptr) primitive_iface->release();
    return success;
}

// primitive_iface_t implementation
dnnl_primitive::dnnl_primitive(
        const std::shared_ptr<primitive_t> &primitive, engine_t *engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<primitive_desc_iface_t>(
              primitive_->pd(), engine)) {}

// reorder specialization
dnnl_primitive::dnnl_primitive(const std::shared_ptr<primitive_t> &primitive,
        engine_t *engine, engine_t *src_engine, engine_t *dst_engine)
    : counter_(1)
    , primitive_(primitive)
    , pd_(utils::make_unique<reorder_primitive_desc_iface_t>(
              primitive_->pd(), engine, src_engine, dst_engine)) {}

dnnl_primitive::~dnnl_primitive() {
    if (scratchpad_debug::is_protect_scratchpad() && scratchpad_ != nullptr
            && scratchpad_->get_memory_storage() != nullptr) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        scratchpad_debug::unprotect_scratchpad_buffer(
                scratchpad_->get_memory_storage(), registry);
    }
}

status_t dnnl_primitive::init() {
    const size_t scratchpad_size
            = primitive_->pd()->scratchpad_size(scratchpad_mode::library);

    if (scratchpad_size) {
        const memory_tracking::registry_t &registry
                = primitive_->pd()->scratchpad_registry();
        bool use_global_scratchpad = scratchpad_debug::is_protect_scratchpad()
                ? false
                : primitive_->use_global_scratchpad();
        auto *scratchpad_ptr = create_scratchpad(
                pd_->engine(), scratchpad_size, use_global_scratchpad);
        if (scratchpad_ptr == nullptr) return out_of_memory;
        if (scratchpad_ptr->get_memory_storage() == nullptr) {
            delete scratchpad_ptr;
            return out_of_memory;
        }

        if (scratchpad_debug::is_protect_scratchpad()) {
            scratchpad_debug::protect_scratchpad_buffer(
                    scratchpad_ptr->get_memory_storage(), registry);
        }
        scratchpad_.reset(scratchpad_ptr);
        if (scratchpad_ptr->size() < scratchpad_size) return out_of_memory;
    }
    return primitive_->create_resource(pd()->engine(), resource_mapper_);
}

engine_t *dnnl_primitive::engine() const {
    return pd_->engine();
}

const primitive_desc_iface_t *dnnl_primitive::pd() const {
    return pd_.get();
}

status_t dnnl_primitive::execute(exec_ctx_t &ctx) const {
    const memory_storage_t *mem_storage = nullptr;
    if (primitive_->pd()->attr()->scratchpad_mode_ == scratchpad_mode::user) {
        memory_t *scratchpad_memory = ctx.output(DNNL_ARG_SCRATCHPAD);
        mem_storage = scratchpad_memory ? scratchpad_memory->memory_storage()
                                        : nullptr;
    } else if (scratchpad_) {
        mem_storage = scratchpad_->get_memory_storage();
    }

    auto scratchpad_grantor
            = primitive_->pd()->scratchpad_registry().grantor(mem_storage, ctx);
    ctx.set_scratchpad_grantor(&scratchpad_grantor);
    ctx.set_resource_mapper(&resource_mapper_);

    auto status = primitive_->execute(ctx);
    ctx.set_scratchpad_grantor(nullptr);
    return status;
}

status_t dnnl_primitive::get_cache_blob_size(size_t *size) const {
    return primitive_->get_cache_blob_size(engine(), size);
}

status_t dnnl_primitive::get_cache_blob(cache_blob_t cache_blob) const {
    return primitive_->get_cache_blob(engine(), cache_blob);
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
