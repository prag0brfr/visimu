// Author: Paulo R. A. Gilz
//
// Persistent thread pool with a parallel_for.
//
// A pool of our own rather than the OS process pool: that would be Win32-only,
// gives no control over width, and pays a dispatch cost per queued item. This
// one is created once, sized to the core count, and lives for the process, so
// loading a column costs zero thread creations.
//
// Built on the primitives in platform_thread.hpp rather than std::thread -- see
// the note there about MinGW's win32 thread model.

#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "aoe/platform_thread.hpp"

namespace aoe {

class ThreadPool {
public:
    explicit ThreadPool(unsigned n_threads = 0) {
        if (n_threads == 0) {
            n_threads = Thread::hardware_concurrency();
            if (n_threads == 0) n_threads = 4;
        }
        // Leave a little headroom on very wide machines; parsing is memory-bound
        // and the extra threads mostly contend for bandwidth.
        if (n_threads > 32) n_threads = 32;

        // unique_ptr because Thread owns an OS handle and is neither copyable
        // nor movable; vector needs one or the other to hold it by value.
        workers_.reserve(n_threads);
        for (unsigned i = 0; i < n_threads; ++i) {
            workers_.emplace_back(new Thread());
            workers_.back()->start(&ThreadPool::worker_entry, this);
        }
    }

    ~ThreadPool() {
        {
            LockGuard guard(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w->join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::size_t size() const noexcept { return workers_.size(); }

    // Runs fn(i) for i in [0, count) across the pool, returning once every call
    // has finished. The calling thread participates, which both uses the core it
    // is sitting on and guarantees forward progress even if every worker is busy.
    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& fn) {
        if (count == 0) return;
        if (count == 1 || workers_.empty()) {
            fn(0);
            return;
        }

        // Shared, not a stack object. The queue keeps one entry per helper and
        // nothing removes the entries nobody claimed -- so when the calling
        // thread drains every index itself (which is the common case for a
        // small job), parallel_for returns and a worker that was still waking
        // up then pops a pointer to a destroyed stack frame and writes to it.
        // A LIFO queue makes that worse: the stale entry can survive to be
        // popped during an entirely unrelated call. Sharing ownership costs one
        // allocation per parallel_for -- nothing beside the work it guards --
        // and makes a late pop harmless, because `next >= count` sends the
        // worker straight back to sleep.
        auto job = std::make_shared<Job>();
        job->fn = &fn;
        job->count = count;
        job->next.store(0, std::memory_order_relaxed);
        job->remaining.store(count, std::memory_order_relaxed);

        std::size_t helpers = workers_.size();
        if (helpers > count - 1) helpers = count - 1;

        {
            LockGuard guard(mutex_);
            for (std::size_t i = 0; i < helpers; ++i) queue_.push_back(job);
        }
        cv_.notify_all();

        run_job(*job);  // caller pulls its share

        // Wait for any indices still in flight on worker threads. `fn` is a
        // reference held by the job, so this must not return early: the
        // caller's callable dies with this frame.
        LockGuard guard(done_mutex_);
        while (job->remaining.load(std::memory_order_acquire) != 0)
            done_cv_.wait(done_mutex_);
    }

    static ThreadPool& global() {
        static ThreadPool pool;
        return pool;
    }

private:
    struct Job {
        const std::function<void(std::size_t)>* fn = nullptr;
        std::size_t count = 0;
        std::atomic<std::size_t> next{0};
        std::atomic<std::size_t> remaining{0};
    };

    void run_job(Job& job) {
        for (;;) {
            std::size_t i = job.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= job.count) break;
            (*job.fn)(i);
            if (job.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                LockGuard guard(done_mutex_);
                done_cv_.notify_all();
            }
        }
    }

    static void worker_entry(void* self) {
        static_cast<ThreadPool*>(self)->worker_loop();
    }

    void worker_loop() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                LockGuard guard(mutex_);
                while (!stopping_ && queue_.empty()) cv_.wait(mutex_);
                if (stopping_ && queue_.empty()) return;
                job = queue_.back();
                queue_.pop_back();
            }
            // The job may already be finished and its parallel_for returned;
            // run_job then finds no index left and drops straight through.
            // Holding a shared_ptr for the duration is what makes that safe.
            run_job(*job);
        }
    }

    std::vector<std::unique_ptr<Thread>> workers_;
    std::vector<std::shared_ptr<Job>> queue_;
    Mutex mutex_;
    CondVar cv_;
    Mutex done_mutex_;
    CondVar done_cv_;
    bool stopping_ = false;
};

}  // namespace aoe
