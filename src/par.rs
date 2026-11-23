use std::sync::atomic::{AtomicUsize, Ordering};

/// parallel map over 0..n with work stealing by chunks; the order of results is kept.
pub fn par_map<T: Send, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    const CHUNK: usize = 32;
    let threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    let next = AtomicUsize::new(0);
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut v = Vec::new();
                    loop {
                        let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= n {
                            break;
                        }
                        for i in start..(start + CHUNK).min(n) {
                            v.push((i, f(i)));
                        }
                    }
                    v
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    for part in parts {
        for (i, t) in part {
            out[i] = Some(t);
        }
    }
    out.into_iter().map(|x| x.unwrap()).collect()
}
