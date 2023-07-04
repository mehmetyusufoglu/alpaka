/* Copyright 2023 Benjamin Worpitz, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/pltf/Traits.hpp"

// Implementation details.
#include "alpaka/acc/AccCpuThreads.hpp"
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/ConcurrentExecPool.hpp"
#include "alpaka/core/Decay.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/workdiv/WorkDivMembers.hpp"

#include <algorithm>
#include <functional>
#include <future>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
#    include <iostream>
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED

namespace alpaka
{
    //! The CPU threads execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelCpuThreads final : public WorkDivMembers<TDim, TIdx>
    {
    private:
        //! The type given to the ConcurrentExecPool for yielding the current thread.
        struct ThreadPoolYield
        {
            //! Yields the current thread.
            ALPAKA_FN_HOST static auto yield() -> void
            {
                std::this_thread::yield();
            }
        };
        // When using the thread pool the threads are yielding because this is faster.
        // Using condition variables and going to sleep is very costly for real threads.
        // Especially when the time to wait is really short (syncBlockThreads) yielding is much faster.
        using ThreadPool = alpaka::core::detail::ConcurrentExecPool<
            TIdx,
            std::thread, // The concurrent execution type.
            std::promise, // The promise type.
            ThreadPoolYield>; // The type yielding the current concurrent execution.

    public:
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelCpuThreads(TWorkDiv&& workDiv, TKernelFnObj const& kernelFnObj, TArgs&&... args)
            : WorkDivMembers<TDim, TIdx>(std::forward<TWorkDiv>(workDiv))
            , m_kernelFnObj(kernelFnObj)
            , m_args(std::forward<TArgs>(
                  args)...) // FIXME(bgruber): this does not forward, since TArgs is not a deduced template parameter
        {
            static_assert(
                Dim<std::decay_t<TWorkDiv>>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //! Executes the kernel function object.
        ALPAKA_FN_HOST auto operator()() const -> void
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            std::apply([&](auto const&... args) { runWithArgs(args...); }, m_args);
        }

    private:
        ALPAKA_FN_HOST auto runWithArgs(std::decay_t<TArgs> const&... args) const -> void
        {
            auto const gridBlockExtent = getWorkDiv<Grid, Blocks>(*this);
            auto const blockThreadExtent = getWorkDiv<Block, Threads>(*this);
            auto const threadElemExtent = getWorkDiv<Thread, Elems>(*this);

            // Get the size of the block shared dynamic memory.
            auto const smBytes = getBlockSharedMemDynSizeBytes<AccCpuThreads<TDim, TIdx>>(
                m_kernelFnObj,
                blockThreadExtent,
                threadElemExtent,
                args...);
#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            std::cout << __func__ << " smBytes: " << smBytes << " B" << std::endl;
#    endif
            AccCpuThreads<TDim, TIdx> acc(*static_cast<WorkDivMembers<TDim, TIdx> const*>(this), smBytes);

            auto const threadsPerBlock = blockThreadExtent.prod();
            ThreadPool threadPool(threadsPerBlock);

            // Execute the blocks serially.
            meta::ndLoopIncIdx(
                gridBlockExtent,
                [&](Vec<TDim, TIdx> const& gridBlockIdx)
                { runBlock(acc, gridBlockIdx, blockThreadExtent, threadPool, m_kernelFnObj, args...); });
        }

        //! The function executed for each grid block.
        ALPAKA_FN_HOST static auto runBlock(
            AccCpuThreads<TDim, TIdx>& acc,
            Vec<TDim, TIdx> const& gridBlockIdx,
            Vec<TDim, TIdx> const& blockThreadExtent,
            ThreadPool& threadPool,
            TKernelFnObj const& kernelFnObj,
            std::decay_t<TArgs> const&... args) -> void
        {
            std::vector<std::future<void>> futuresInBlock;
            acc.m_gridBlockIdx = gridBlockIdx;

            // Execute the threads of this block in parallel.
            meta::ndLoopIncIdx(
                blockThreadExtent,
                [&](Vec<TDim, TIdx> const& blockThreadIdx)
                {
                    // copy blockThreadIdx because it will get changed for the next iteration/thread.
                    futuresInBlock.emplace_back(threadPool.enqueueTask(
                        [&, blockThreadIdx] { runThread(acc, blockThreadIdx, kernelFnObj, args...); }));
                });

            // Wait for the completion of the block thread kernels.
            for(auto& t : futuresInBlock)
                t.wait();

            // Clean up.
            futuresInBlock.clear();
            acc.m_threadToIndexMap.clear();
            freeSharedVars(acc); // After a block has been processed, the shared memory has to be deleted.
        }

        //! The thread entry point on the accelerator.
        ALPAKA_FN_HOST static auto runThread(
            AccCpuThreads<TDim, TIdx>& acc,
            Vec<TDim, TIdx> const& blockThreadIdx,
            TKernelFnObj const& kernelFnObj,
            std::decay_t<TArgs> const&... args) -> void
        {
            // We have to store the thread data before the kernel is calling any of the methods of this class depending
            // on them.
            auto const threadId = std::this_thread::get_id();

            if(blockThreadIdx.sum() == 0)
            {
                acc.m_idMasterThread = threadId;
            }

            {
                // Save the thread id, and index.
                std::lock_guard<std::mutex> lock(acc.m_mtxMapInsert);
                acc.m_threadToIndexMap.emplace(threadId, blockThreadIdx);
            }

            // Sync all threads so that the maps with thread id's are complete and not changed after here.
            syncBlockThreads(acc);

            // Execute the kernel itself.
            kernelFnObj(std::as_const(acc), args...);

            // We have to sync all threads here because if a thread would finish before all threads have been started,
            // a new thread could get the recycled (then duplicate) thread id!
            syncBlockThreads(acc);
        }

        TKernelFnObj m_kernelFnObj;
        std::tuple<std::decay_t<TArgs>...> m_args;
    };

    namespace trait
    {
        //! The CPU threads execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccCpuThreads<TDim, TIdx>;
        };

        //! The CPU threads execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevCpu;
        };

        //! The CPU threads execution task dimension getter trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DimType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TDim;
        };

        //! The CPU threads execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfCpu;
        };

        //! The CPU threads execution task idx type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct IdxType<TaskKernelCpuThreads<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = TIdx;
        };
    } // namespace trait
} // namespace alpaka

#endif
