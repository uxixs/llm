//! Persistent thread pool, std only (no rayon / crossbeam).
//!
//! `ThreadPool::new(n)` spawns n-1 workers parked on a `Condvar`; the thread
//! calling `run` is the n-th executor. `run(n_tasks, f)` hands out task
//! indices 0..n_tasks through an atomic counter, so each index runs exactly
//! once, and blocks until all of them finished.
//!
//! # State machine (all fields of `State` change under `Shared::state`)
//! - Idle: `job == None`. Workers sleep on `work_cv`.
//! - Publish: the caller resets `next` and `panic`, stores `job` and
//!   `n_tasks`, bumps `epoch`, and notifies all workers.
//! - Join: a worker that sees a new epoch with `job == Some` increments
//!   `active` and copies the job out (under the lock); it then claims indices
//!   from `next` until they run out, and finally decrements `active` (under
//!   the lock, notifying `done_cv` at 0). A worker that wakes after the job
//!   was cleared goes back to sleep.
//! - Retire: the caller, after running indices itself, waits on `done_cv`
//!   until `active == 0` and clears `job` in that same critical section, so
//!   no worker can join between "all idle" and "job gone".
//!
//! # Panics (Ruling K)
//! Every task call is wrapped in `catch_unwind`. A panicking task records the
//! first payload; the remaining indices are still drained (never skipped). The
//! retire step runs in a drop guard, so `run` cannot return or unwind before
//! every worker has stopped touching `f`; only then is the first panic
//! re-raised on the caller with `resume_unwind`. The pool stays usable.
//!
//! # Reentrancy
//! A thread that is currently executing a task of this pool (a worker, or the
//! caller inside `run`) and calls `run` on the same pool executes the nested
//! run inline, sequentially, on that thread (panics then propagate directly).
//! Concurrent `run` calls from unrelated threads are serialized by a mutex.
//! Cross-pool cycles (a task of pool A runs pool B whose task runs pool A) are
//! not detected and can deadlock; don't build those.

use std::any::Any;
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;

type Task = dyn Fn(usize) + Sync;
type Payload = Box<dyn Any + Send + 'static>;

/// Lifetime-erased pointer to the current run's closure.
#[derive(Clone, Copy)]
struct JobPtr(*const Task);
// SAFETY: the pointee is `dyn Fn + Sync`, so calling it from any thread is
// fine; the pointer is only dereferenced while the owning `run` call is
// blocked in `RetireGuard` waiting for `active == 0` (see `ThreadPool::run`).
unsafe impl Send for JobPtr {}

struct State {
    job: Option<JobPtr>,
    n_tasks: usize,
    epoch: u64,
    active: usize,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    work_cv: Condvar,
    done_cv: Condvar,
    next: AtomicUsize,
    panic: Mutex<Option<Payload>>,
}

thread_local! {
    /// The pool whose task this thread is currently executing (null if none).
    static CURRENT: Cell<*const Shared> = const { Cell::new(std::ptr::null()) };
}

/// Lock ignoring poison: nothing panics while holding these locks (tasks run
/// unlocked), and the protected data stays consistent regardless.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Shared {
    /// Claims and runs indices until exhausted, catching task panics.
    fn drain(&self, f: &(dyn Fn(usize) + Sync + '_), n_tasks: usize) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            if let Err(p) = panic::catch_unwind(AssertUnwindSafe(|| f(i))) {
                let mut slot = lock(&self.panic);
                if slot.is_none() {
                    *slot = Some(p);
                }
            }
        }
    }
}

/// Restores the previous `CURRENT` on drop.
struct CurrentGuard(*const Shared);
impl Drop for CurrentGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(self.0));
    }
}

fn enter(shared: *const Shared) -> CurrentGuard {
    CurrentGuard(CURRENT.with(|c| c.replace(shared)))
}

/// Retire step of `run`: waits until no worker holds the job, then clears it.
/// Runs on drop, so it also happens if anything between publish and the
/// normal end of `run` unwinds.
struct RetireGuard<'a>(&'a Shared);
impl Drop for RetireGuard<'_> {
    fn drop(&mut self) {
        let mut st = lock(&self.0.state);
        while st.active > 0 {
            st = self.0.done_cv.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        st.job = None;
    }
}

pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Serializes `run` calls from unrelated threads.
    run_lock: Mutex<()>,
    n: usize,
}

impl ThreadPool {
    /// Pool with `n` executors: n-1 spawned workers plus the calling thread.
    pub fn new(n: usize) -> Self {
        assert!(n >= 1, "ThreadPool needs n >= 1 (got {n})");
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                job: None,
                n_tasks: 0,
                epoch: 0,
                active: 0,
                shutdown: false,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            next: AtomicUsize::new(0),
            panic: Mutex::new(None),
        });
        let workers = (1..n)
            .map(|i| {
                let sh = Arc::clone(&shared);
                std::thread::Builder::new()
                    .name(format!("rllm-pool-{i}"))
                    .spawn(move || worker(sh))
                    .expect("spawn pool worker")
            })
            .collect();
        ThreadPool {
            shared,
            workers,
            run_lock: Mutex::new(()),
            n,
        }
    }

    /// Number of executors (spawned workers + the calling thread).
    pub fn n_threads(&self) -> usize {
        self.n
    }

    /// Runs `f(i)` exactly once for every i in 0..n_tasks across the pool and
    /// blocks until all calls finished. Re-raises the first task panic after
    /// every task has finished. Nested calls from inside a task of this pool
    /// run inline on the calling thread.
    pub fn run(&self, n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        if n_tasks == 0 {
            return;
        }
        let me: *const Shared = &*self.shared;
        let nested = CURRENT.with(|c| c.get()) == me;
        if nested || self.workers.is_empty() || n_tasks == 1 {
            // Inline: no other thread sees `f`, panics propagate directly.
            let _cur = enter(me);
            for i in 0..n_tasks {
                f(i);
            }
            return;
        }
        let _serial = lock(&self.run_lock);
        // SAFETY (Ruling K): this erases the lifetime of `f` so workers can
        // call it. A worker obtains the pointer only by joining the current
        // epoch under `state` (incrementing `active`) and stops using it
        // before decrementing `active`. `RetireGuard`, which is dropped before
        // this function returns and also if it unwinds, waits for
        // `active == 0` and clears `job` in the same critical section, so
        // afterwards no worker holds or can obtain the pointer. Task panics
        // are caught in `drain`, so nothing unwinds past the guard early; the
        // first panic is re-raised only after the guard ran. Hence `f`
        // outlives every use of the pointer.
        let job = JobPtr(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const Task>(f)
        });
        // Armed before publishing, so the job is never visible unguarded.
        let retire = RetireGuard(&self.shared);
        {
            let mut st = lock(&self.shared.state);
            self.shared.next.store(0, Ordering::Relaxed);
            *lock(&self.shared.panic) = None;
            st.job = Some(job);
            st.n_tasks = n_tasks;
            st.epoch = st.epoch.wrapping_add(1);
            self.shared.work_cv.notify_all();
        }
        {
            let _cur = enter(me);
            self.shared.drain(f, n_tasks);
        }
        drop(retire);
        let payload = lock(&self.shared.panic).take();
        if let Some(p) = payload {
            panic::resume_unwind(p);
        }
    }
}

fn worker(sh: Arc<Shared>) {
    // Workers only ever execute tasks of this pool.
    CURRENT.with(|c| c.set(&*sh));
    let mut seen = 0u64;
    loop {
        let (job, n_tasks) = {
            let mut st = lock(&sh.state);
            loop {
                if st.shutdown {
                    return;
                }
                if st.epoch != seen {
                    if let Some(j) = st.job {
                        seen = st.epoch;
                        st.active += 1;
                        break (j, st.n_tasks);
                    }
                }
                st = sh.work_cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        };
        // SAFETY: we joined this epoch (active += 1) while `job` was set, so
        // the owning `run` stays blocked in `RetireGuard` until we decrement
        // `active` below; the closure is alive for this whole block, and
        // `drain` catches task panics so we always reach the decrement.
        let f: &Task = unsafe { &*job.0 };
        sh.drain(f, n_tasks);
        let mut st = lock(&sh.state);
        st.active -= 1;
        if st.active == 0 {
            sh.done_cv.notify_all();
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        lock(&self.shared.state).shutdown = true;
        self.shared.work_cv.notify_all();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn pool_runs_all_tasks_once() {
        for &size in &[1usize, 4] {
            let pool = ThreadPool::new(size);
            assert_eq!(pool.n_threads(), size);
            for &n_tasks in &[0usize, 1, 3, 100] {
                let hits: Vec<AtomicUsize> = (0..n_tasks).map(|_| AtomicUsize::new(0)).collect();
                pool.run(n_tasks, &|i| {
                    hits[i].fetch_add(1, Ordering::Relaxed);
                });
                for (i, h) in hits.iter().enumerate() {
                    assert_eq!(
                        h.load(Ordering::Relaxed),
                        1,
                        "size {size} n {n_tasks} idx {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn pool_reusable_1000_runs() {
        let pool = ThreadPool::new(4);
        let total = AtomicUsize::new(0);
        for r in 0..1000 {
            let n_tasks = r % 17;
            pool.run(n_tasks, &|i| {
                total.fetch_add(i + 1, Ordering::Relaxed);
            });
        }
        let want: usize = (0..1000).map(|r| (r % 17) * (r % 17 + 1) / 2).sum();
        assert_eq!(total.load(Ordering::Relaxed), want);
    }

    #[test]
    #[should_panic(expected = "task 5 failed")]
    fn pool_propagates_panic() {
        let pool = ThreadPool::new(4);
        pool.run(16, &|i| {
            if i == 5 {
                panic!("task 5 failed");
            }
        });
    }

    /// Ruling K: run must not return/unwind while any worker still touches f
    /// (which borrows `written` on this stack frame), and the pool survives.
    #[test]
    fn pool_panic_waits_for_all_tasks() {
        let pool = ThreadPool::new(4);
        let written: Vec<AtomicUsize> = (0..8).map(|_| AtomicUsize::new(usize::MAX)).collect();
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(8, &|i| {
                if i == 0 {
                    panic!("task 0 boom");
                }
                std::thread::sleep(Duration::from_millis(20));
                written[i].store(i, Ordering::SeqCst);
            })
        }));
        let err = r.expect_err("run must re-raise the task panic");
        assert_eq!(err.downcast_ref::<&str>().copied(), Some("task 0 boom"));
        // The pool drains (never skips) the remaining tasks after a panic, and
        // run unwinds only after all of them finished: all 7 writes landed.
        for (i, w) in written.iter().enumerate().skip(1) {
            assert_eq!(w.load(Ordering::SeqCst), i, "task {i} write missing");
        }
        // Pool still usable.
        let hits: Vec<AtomicUsize> = (0..50).map(|_| AtomicUsize::new(0)).collect();
        pool.run(50, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// Nested run on the same pool from inside a task executes inline.
    #[test]
    fn pool_nested_run_inline() {
        let pool = ThreadPool::new(4);
        let hits: Vec<AtomicUsize> = (0..8 * 5).map(|_| AtomicUsize::new(0)).collect();
        pool.run(8, &|i| {
            pool.run(5, &|j| {
                hits[i * 5 + j].fetch_add(1, Ordering::Relaxed);
            });
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    /// Concurrent callers from different threads are serialized, not mixed.
    #[test]
    fn pool_concurrent_callers() {
        let pool = ThreadPool::new(3);
        let hits: Vec<AtomicUsize> = (0..4 * 64).map(|_| AtomicUsize::new(0)).collect();
        std::thread::scope(|s| {
            for c in 0..4 {
                let (pool, hits) = (&pool, &hits);
                s.spawn(move || {
                    for _ in 0..50 {
                        pool.run(64, &|i| {
                            hits[c * 64 + i].fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 50));
    }
}
