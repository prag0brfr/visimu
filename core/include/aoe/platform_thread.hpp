// Author: Paulo R. A. Gilz
//
// Thin threading primitives over Win32 / pthreads.
//
// Why not std::thread: MinGW-w64 ships in two flavours, and the one built with
// the *win32* thread model has no working <thread>, <mutex> or
// <condition_variable> -- they either fail to compile or link against missing
// gthread symbols. The target toolchain here is a portable MinGW-w64 GCC 10.2
// whose thread model we cannot verify in advance, so relying on std::thread
// would risk shipping code that simply does not link on the one machine that
// matters.
//
// Going native also happens to be faster on Windows: SRWLOCK and
// CONDITION_VARIABLE are lighter than libstdc++'s winpthreads-backed wrappers.
// On POSIX these map to pthreads with no measurable difference from std::thread.

#pragma once

#include <cstddef>

#if defined(_WIN32)
// Brings in <windows.h> with _WIN32_WINNT set. The SRWLOCK and
// CONDITION_VARIABLE types below are declared only when it is at least 0x0600,
// so including <windows.h> directly here makes this file's errors depend on
// whatever default the toolchain happens to carry.
#include "aoe/win32.hpp"
#else
#include <pthread.h>
#include <unistd.h>
#endif

namespace aoe {

// ---------------------------------------------------------------- Mutex

class Mutex {
public:
#if defined(_WIN32)
    Mutex() noexcept { ::InitializeSRWLock(&lock_); }
    ~Mutex() = default;
    void lock() noexcept { ::AcquireSRWLockExclusive(&lock_); }
    void unlock() noexcept { ::ReleaseSRWLockExclusive(&lock_); }
    SRWLOCK* native() noexcept { return &lock_; }
#else
    Mutex() noexcept { ::pthread_mutex_init(&lock_, nullptr); }
    ~Mutex() { ::pthread_mutex_destroy(&lock_); }
    void lock() noexcept { ::pthread_mutex_lock(&lock_); }
    void unlock() noexcept { ::pthread_mutex_unlock(&lock_); }
    pthread_mutex_t* native() noexcept { return &lock_; }
#endif

    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;

private:
#if defined(_WIN32)
    SRWLOCK lock_;
#else
    pthread_mutex_t lock_;
#endif
};

class LockGuard {
public:
    explicit LockGuard(Mutex& m) noexcept : m_(m) { m_.lock(); }
    ~LockGuard() { m_.unlock(); }
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    Mutex& m_;
};

// --------------------------------------------------------------- RwLock
//
// Many concurrent readers, one exclusive writer.
//
// The front end reads the session from the GUI thread -- headers, parameter
// lists, cached channels, the memory total -- while a worker is inside it
// opening files or reloading one. Those are structural changes: add_file grows
// a vector of files, clear() destroys them, reload() replaces a dataset's
// simulations. A reader walking that list while it reallocates is reading freed
// memory, which is what a crash while ticking runs on and off actually was.
//
// A plain mutex would serialise the reads against each other too, and the reads
// are the common case by a wide margin. SRWLOCK and pthread_rwlock both give
// shared acquisition for nothing extra.

class RwLock {
public:
#if defined(_WIN32)
    RwLock() noexcept { ::InitializeSRWLock(&lock_); }
    ~RwLock() = default;
    void lock() noexcept { ::AcquireSRWLockExclusive(&lock_); }
    void unlock() noexcept { ::ReleaseSRWLockExclusive(&lock_); }
    void lock_shared() noexcept { ::AcquireSRWLockShared(&lock_); }
    void unlock_shared() noexcept { ::ReleaseSRWLockShared(&lock_); }
#else
    RwLock() noexcept { ::pthread_rwlock_init(&lock_, nullptr); }
    ~RwLock() { ::pthread_rwlock_destroy(&lock_); }
    void lock() noexcept { ::pthread_rwlock_wrlock(&lock_); }
    void unlock() noexcept { ::pthread_rwlock_unlock(&lock_); }
    void lock_shared() noexcept { ::pthread_rwlock_rdlock(&lock_); }
    void unlock_shared() noexcept { ::pthread_rwlock_unlock(&lock_); }
#endif

    RwLock(const RwLock&) = delete;
    RwLock& operator=(const RwLock&) = delete;

private:
#if defined(_WIN32)
    SRWLOCK lock_;
#else
    pthread_rwlock_t lock_;
#endif
};

class WriteGuard {
public:
    explicit WriteGuard(RwLock& l) noexcept : lock_(l) { lock_.lock(); }
    ~WriteGuard() { lock_.unlock(); }
    WriteGuard(const WriteGuard&) = delete;
    WriteGuard& operator=(const WriteGuard&) = delete;

private:
    RwLock& lock_;
};

class ReadGuard {
public:
    explicit ReadGuard(RwLock& l) noexcept : lock_(l) { lock_.lock_shared(); }
    ~ReadGuard() { lock_.unlock_shared(); }
    ReadGuard(const ReadGuard&) = delete;
    ReadGuard& operator=(const ReadGuard&) = delete;

private:
    RwLock& lock_;
};

// ------------------------------------------------------------ CondVar

class CondVar {
public:
#if defined(_WIN32)
    CondVar() noexcept { ::InitializeConditionVariable(&cv_); }
    ~CondVar() = default;
    void wait(Mutex& m) noexcept {
        ::SleepConditionVariableSRW(&cv_, m.native(), INFINITE, 0);
    }
    void notify_one() noexcept { ::WakeConditionVariable(&cv_); }
    void notify_all() noexcept { ::WakeAllConditionVariable(&cv_); }
#else
    CondVar() noexcept { ::pthread_cond_init(&cv_, nullptr); }
    ~CondVar() { ::pthread_cond_destroy(&cv_); }
    void wait(Mutex& m) noexcept { ::pthread_cond_wait(&cv_, m.native()); }
    void notify_one() noexcept { ::pthread_cond_signal(&cv_); }
    void notify_all() noexcept { ::pthread_cond_broadcast(&cv_); }
#endif

    CondVar(const CondVar&) = delete;
    CondVar& operator=(const CondVar&) = delete;

private:
#if defined(_WIN32)
    CONDITION_VARIABLE cv_;
#else
    pthread_cond_t cv_;
#endif
};

// ------------------------------------------------------------- Thread

class Thread {
public:
    using EntryFn = void (*)(void*);

    Thread() = default;
    ~Thread() { join(); }

    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    bool start(EntryFn fn, void* arg) noexcept {
        entry_ = fn;
        arg_ = arg;
#if defined(_WIN32)
        handle_ = ::CreateThread(nullptr, 0, &Thread::trampoline, this, 0, nullptr);
        return handle_ != nullptr;
#else
        return ::pthread_create(&handle_, nullptr, &Thread::trampoline, this) == 0
                   ? (started_ = true)
                   : false;
#endif
    }

    void join() noexcept {
#if defined(_WIN32)
        if (handle_) {
            ::WaitForSingleObject(handle_, INFINITE);
            ::CloseHandle(handle_);
            handle_ = nullptr;
        }
#else
        if (started_) {
            ::pthread_join(handle_, nullptr);
            started_ = false;
        }
#endif
    }

    static unsigned hardware_concurrency() noexcept {
#if defined(_WIN32)
        SYSTEM_INFO info;
        ::GetSystemInfo(&info);
        return info.dwNumberOfProcessors > 0 ? info.dwNumberOfProcessors : 1u;
#else
        long n = ::sysconf(_SC_NPROCESSORS_ONLN);
        return n > 0 ? static_cast<unsigned>(n) : 1u;
#endif
    }

private:
#if defined(_WIN32)
    static DWORD WINAPI trampoline(LPVOID self) {
        Thread* t = static_cast<Thread*>(self);
        t->entry_(t->arg_);
        return 0;
    }
    HANDLE handle_ = nullptr;
#else
    static void* trampoline(void* self) {
        Thread* t = static_cast<Thread*>(self);
        t->entry_(t->arg_);
        return nullptr;
    }
    pthread_t handle_{};
    bool started_ = false;
#endif
    EntryFn entry_ = nullptr;
    void* arg_ = nullptr;
};

}  // namespace aoe
