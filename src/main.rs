mod bench;
mod cache;
mod deps;
mod gpu;
mod gpu_krc;
mod grid;
mod image;
mod math;
mod methods;
mod par;
mod pt;
mod render;
mod rt;
mod scene;
mod tracer;

use render::{gbuffer, load_raw, reference, save_raw, World};
use scene::{build, state_at, FRAMES};
use std::collections::HashMap;
use std::time::Instant;

fn ensure_refs(world: &World, dir: &str, tracer: &tracer::Tracer) -> HashMap<String, Vec<math::V3>> {
    std::fs::create_dir_all(dir).unwrap();
    let mut refs = HashMap::new();
    for f in 0..FRAMES {
        let s = state_at(f);
        let id = s.id();
        if refs.contains_key(&id) {
            continue;
        }
        let path = format!("{dir}/{id}.bin");
        let img = load_raw(&path, world.len()).unwrap_or_else(|| {
            let t = Instant::now();
            let scene = build(&s, world.rooms);
            let img = reference(&scene, &gbuffer(&scene, world), tracer);
            save_raw(&path, &img);
            image::save(&format!("{dir}/{id}.png"), world.w, world.h, &img);
            eprintln!("reference {dir}/{id}: {:.1} s", t.elapsed().as_secs_f64());
            img
        });
        refs.insert(id, img);
    }
    refs
}

/// Phases of the sequence: (name, first frame, last frame excluded).
const PHASES: [(&str, usize, usize); 6] = [
    ("all", 0, 120),
    ("static", 30, 40),
    ("box", 40, 64),
    ("after_box", 64, 80),
    ("lamp_on", 80, 100),
    ("lamp_off", 100, 120),
];

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = if v.len() > 1 { v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0) } else { 0.0 };
    (m, var.sqrt())
}

fn run(
    names: &[String],
    seeds: u64,
    refs: &HashMap<String, Vec<math::V3>>,
    world: &World,
    root: &str,
    warmup: u32,
    tracer: &tracer::Tracer,
) -> String {
    let mut all = methods::all_methods();
    if !names.is_empty() {
        all.retain(|c| names.contains(&c.name));
    }
    if !tracer.is_gpu() {
        all.retain(|c| !c.name.starts_with("gpu_"));
    }
    let mut summary = String::from("method,phase,relmse_mean,relmse_std,rmse_mean,paths_per_frame\n");
    for cfg in &all {
        let t = Instant::now();
        let dir = format!("{root}/{}", cfg.name);
        let mut runs = Vec::new();
        for seed in 0..seeds {
            let stats = methods::run(cfg, refs, &dir, seed, world, warmup, tracer);
            let mut csv = String::from(methods::FrameStats::HEADER);
            csv.push('\n');
            for s in &stats {
                csv += &s.csv();
                csv.push('\n');
            }
            let name = if seed == 0 { "metrics.csv".to_string() } else { format!("metrics_s{seed}.csv") };
            std::fs::write(format!("{dir}/{name}"), csv).unwrap();
            runs.push(stats);
        }
        let mut line = format!("{:<18}", cfg.name);
        for (phase, a, b) in PHASES {
            let per_run = |f: &dyn Fn(&methods::FrameStats) -> f64| -> Vec<f64> {
                runs.iter().map(|r| r[a..b].iter().map(f).sum::<f64>() / (b - a) as f64).collect()
            };
            let (rel_m, rel_s) = mean_std(&per_run(&|s| s.relmse));
            let (rmse_m, _) = mean_std(&per_run(&|s| s.rmse));
            let (paths_m, _) = mean_std(&per_run(&|s| s.paths as f64));
            summary += &format!("{},{phase},{rel_m:.6e},{rel_s:.6e},{rmse_m:.6e},{paths_m:.1}\n", cfg.name);
            line += &format!(" | {phase} {rel_m:.2e}±{rel_s:.1e}");
        }
        eprintln!("{line} | {:.1} s", t.elapsed().as_secs_f64());
    }
    let path = if names.is_empty() { format!("{root}/summary.csv") } else { format!("{root}/summary_partial.csv") };
    std::fs::write(path, &summary).unwrap();
    summary
}

/// Scaling: N rooms, same events (rooms 0 and 1), constant pixel density.
fn scale(tracer: &tracer::Tracer) {
    let only: Option<Vec<String>> = std::env::var("KRC_METHODS").ok().map(|v| v.split(',').map(String::from).collect());
    let names: Vec<String> = [
        "cache_reset",
        "krc_hard",
        "krc_all",
        "krc_noage",
        "krc",
        "cache_reset_target",
        "krc_hard_target",
        "krc_all_target",
        "krc_noage_target",
        "krc_target",
        "krc_energy",
        "krc_energy_target",
        "krc_energy_l1",
        "krc_energy_l2",
        "krc_energy_l2_target",
    ]
    .iter()
    .map(|s| s.to_string())
    .filter(|s| only.as_ref().map_or(true, |o| o.contains(s)))
    .collect();
    let mut all = String::from("rooms,method,phase,relmse_mean,relmse_std,rmse_mean,paths_per_frame\n");
    let rooms_list: Vec<usize> = std::env::var("KRC_ROOMS")
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|_| vec![2, 4, 8]);
    for rooms in rooms_list {
        let world = World::new(rooms);
        let refs = ensure_refs(&world, &format!("results/ref_n{rooms}"), tracer);
        eprintln!("--- {rooms} rooms");
        let summary = run(&names, seeds(), &refs, &world, &format!("{}/scale/n{rooms}", out_root(tracer)), warmup(), tracer);
        for line in summary.lines().skip(1) {
            all += &format!("{rooms},{line}\n");
        }
    }
    if only.is_none() {
        std::fs::write(format!("{}/scale/summary.csv", out_root(tracer)), all).unwrap();
    }
}

/// Number of independent seeds per method (KRC_SEEDS environment variable, 5 by default).
fn seeds() -> u64 {
    std::env::var("KRC_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5)
}

/// Warm-up frames before the scaling sequence (KRC_WARMUP, 150 by default):
/// the measured costs are then those of the events, not of the initial convergence.
fn warmup() -> u32 {
    std::env::var("KRC_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(150)
}

/// Results folder: `results` (CPU), `results/gpu` (GPU), or KRC_OUT.
fn out_root(tracer: &tracer::Tracer) -> String {
    std::env::var("KRC_OUT").unwrap_or_else(|_| if tracer.is_gpu() { "results/gpu".into() } else { "results".into() })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tracer = tracer::Tracer::from_env();
    let w2 = World::new(2);
    match args.first().map(String::as_str) {
        Some("refs") => {
            ensure_refs(&w2, "results/ref", &tracer);
        }
        Some("run") => {
            run(&args[1..], seeds(), &ensure_refs(&w2, "results/ref", &tracer), &w2, &out_root(&tracer), 0, &tracer);
        }
        Some("all") => {
            run(&[], seeds(), &ensure_refs(&w2, "results/ref", &tracer), &w2, &out_root(&tracer), 0, &tracer);
        }
        Some("scale") => scale(&tracer),
        Some("bench") => bench::bench(&tracer),
        Some("drift") => match gpu::Gpu::new() {
            Some(g) => {
                let cycles = args.get(1).and_then(|c| c.parse().ok()).unwrap_or(100);
                gpu_krc::drift(&g, cycles, &ensure_refs(&w2, "results/ref", &tracer));
            }
            None => eprintln!("no GPU"),
        },
        Some("cornellcam") => match gpu::Gpu::new() {
            Some(g) => {
                let size = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
                rt::run_tri(&g, seeds(), &rt::cornell_cam_setup(size));
            }
            None => eprintln!("no GPU"),
        },
        Some("sponzacam") => match gpu::Gpu::new() {
            Some(g) => {
                let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(480);
                rt::run_tri(&g, seeds(), &rt::sponza_cam_setup(w, w * 9 / 16));
            }
            None => eprintln!("no GPU"),
        },
        Some("cornell") => match gpu::Gpu::new() {
            Some(g) => {
                let size = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
                rt::run_tri(&g, seeds(), &rt::cornell_setup(size));
            }
            None => eprintln!("no GPU"),
        },
        Some("sponza") => match gpu::Gpu::new() {
            Some(g) => {
                let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(480);
                rt::run_tri(&g, seeds(), &rt::sponza_setup(w, w * 9 / 16));
            }
            None => eprintln!("no GPU"),
        },
        Some("pareto") => match gpu::Gpu::new() {
            Some(g) => match args.get(1).map(String::as_str) {
                Some("sponza") => {
                    let w: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1280);
                    rt::pareto(&g, &rt::sponza_setup(w, w * 9 / 16))
                }
                Some("sponzacam") => {
                    let w: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1280);
                    rt::pareto(&g, &rt::sponza_cam_setup(w, w * 9 / 16))
                }
                Some("cornellcam") => {
                    rt::pareto(&g, &rt::cornell_cam_setup(args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024)))
                }
                _ => rt::pareto(&g, &rt::cornell_setup(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024))),
            },
            None => eprintln!("no GPU"),
        },
        Some("spikes") => match gpu::Gpu::new() {
            Some(g) => match args.get(1).map(String::as_str) {
                Some("sponza") => rt::spikes(&g, &rt::sponza_setup(1280, 720)),
                _ => rt::spikes(&g, &rt::cornell_setup(args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024))),
            },
            None => eprintln!("no GPU"),
        },
        Some("bias") => match gpu::Gpu::new() {
            Some(g) => match args.get(1).map(String::as_str) {
                Some("cornell") => rt::bias(&g, &rt::cornell_setup(256)),
                Some("cornellcam") => rt::bias(&g, &rt::cornell_cam_setup(256)),
                Some("sponzacam") => rt::bias(&g, &rt::sponza_cam_setup(480, 270)),
                _ => rt::bias(&g, &rt::sponza_setup(480, 270)),
            },
            None => eprintln!("no GPU"),
        },
        Some("game") => match gpu::Gpu::new() {
            Some(g) => {
                let setup = match args.get(1).map(String::as_str) {
                    Some("sponza") => rt::sponza_setup(480, 270),
                    Some("cornellcam") => rt::cornell_cam_setup(256),
                    Some("cornell") => rt::cornell_setup(256),
                    Some("sponzacam720") => rt::sponza_cam_setup(1280, 720),
                    _ => rt::sponza_cam_setup(480, 270),
                };
                let only = std::env::var("KRC_DUEL").ok();
                let cfgs: Vec<(rt::DuelCfg, bool)> = game_cfgs()
                    .into_iter()
                    .filter(|(c, _)| only.as_ref().is_none_or(|o| o.split(',').any(|x| c.name.starts_with(x))))
                    .collect();
                rt::game_eval(&g, &setup, &cfgs);
            }
            None => eprintln!("no GPU"),
        },
        Some("duel") | Some("static") => match gpu::Gpu::new() {
            Some(g) => {
                let setup = match args.get(1).map(String::as_str) {
                    Some("sponza") => rt::sponza_setup(480, 270),
                    Some("cornellcam") => rt::cornell_cam_setup(256),
                    Some("cornell") => rt::cornell_setup(256),
                    _ => rt::sponza_cam_setup(480, 270),
                };
                let only = std::env::var("KRC_DUEL").ok();
                let cfgs: Vec<rt::DuelCfg> =
                    duel_cfgs().into_iter().filter(|c| only.as_ref().is_none_or(|o| o.split(',').any(|x| c.name.starts_with(x)))).collect();
                if args[0] == "duel" {
                    rt::duel(&g, &setup, &cfgs);
                } else {
                    rt::static_conv(&g, &setup, &cfgs);
                }
            }
            None => eprintln!("no GPU"),
        },
        Some("passes") => match gpu::Gpu::new() {
            Some(g) => {
                let ppp = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0);
                let setup = match args.get(1).map(String::as_str) {
                    Some("sponza") => rt::sponza_setup(1280, 720),
                    Some("cornellcam") => rt::cornell_cam_setup(1024),
                    Some("sponzacam480") => rt::sponza_cam_setup(480, 270),
                    Some("showcase") => rt::showcase_setup(960, 540),
                    _ => rt::sponza_cam_setup(1280, 720),
                };
                rt::passes(&g, &setup, ppp);
            }
            None => eprintln!("no GPU"),
        },
        Some("showcase") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_video(&g, args.get(1).and_then(|s| s.parse().ok()).unwrap_or(960)),
            None => eprintln!("no GPU"),
        },
        Some("showcaseref") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_ref(
                &g,
                args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60),
                args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64),
            ),
            None => eprintln!("no GPU"),
        },
        Some("showcasedbg") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_dbg(&g, args.get(1).and_then(|s| s.parse().ok()).unwrap_or(39)),
            None => eprintln!("no GPU"),
        },
        Some("showcaselum") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_lum(&g),
            None => eprintln!("no GPU"),
        },
        Some("showcaseregref") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_regref(&g, args.get(1).and_then(|s| s.parse().ok()).unwrap_or(39)),
            None => eprintln!("no GPU"),
        },
        Some("showcasems") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_ms(&g, args.get(1).and_then(|s| s.parse().ok()).unwrap_or(960)),
            None => eprintln!("no GPU"),
        },
        Some("showcasetest") => match gpu::Gpu::new() {
            Some(g) => rt::showcase_test(&g, args.get(1).and_then(|s| s.parse().ok()).unwrap_or(960)),
            None => eprintln!("no GPU"),
        },
        Some("sharccheck") => match gpu::Gpu::new() {
            Some(g) => match args.get(1).map(String::as_str) {
                Some("sponza") => rt::sharc_check(&g, &rt::sponza_setup(480, 270)),
                _ => rt::sharc_check(&g, &rt::cornell_setup(256)),
            },
            None => eprintln!("no GPU"),
        },
        Some("sponzadebug") => match gpu::Gpu::new() {
            Some(g) => rt::sponza_debug(&g),
            None => eprintln!("no GPU"),
        },
        Some("rtsmoke") => match gpu::Gpu::new() {
            Some(g) => rt::smoke(&g),
            None => eprintln!("no GPU"),
        },
        Some("gpubench") => match gpu::Gpu::new() {
            Some(g) => gpu_krc::bench(&g),
            None => eprintln!("no GPU"),
        },
        _ => eprintln!("usage: krc refs | run [methods...] | all | scale | bench"),
    }
}

/// Configurations of the KRC / SHaRC duel (filtered by KRC_DUEL, comma-separated prefixes).
fn duel_cfgs() -> Vec<rt::DuelCfg> {
    use gpu_krc::{Method, ShParams};
    let d = ShParams::default();
    let krcs = |name, ppp, resample, cell_px, reuse, demod, seed| rt::DuelCfg {
        name,
        method: Method::Krc,
        ppp,
        resample,
        cell_px,
        reuse,
        demod,
        seed,
        support: false,
        tau: 0.0,
        multi: false,
        sh: d,
        m_light: 0,
        debt_rate: -1.0,
        reset_ratio: 0.0,
        dn_gamma: 0.0,
        drs: false,
    };
    let krcd = |name, ppp, resample, cell_px, reuse, demod| krcs(name, ppp, resample, cell_px, reuse, demod, false);
    let krcr = |name, ppp, resample, cell_px, reuse| krcd(name, ppp, resample, cell_px, reuse, false);
    let krcc = |name, ppp, resample, cell_px| krcr(name, ppp, resample, cell_px, false);
    let krc = |name, ppp, resample| krcc(name, ppp, resample, 4.0);
    let nv = |name, down| rt::DuelCfg {
        name,
        method: Method::Sharc,
        ppp: 1.0,
        resample: false,
        cell_px: 4.0,
        reuse: false,
        demod: false,
        seed: false,
        support: false,
        tau: 0.0,
        multi: false,
        sh: ShParams { downscale: down, ..d },
        ..Default::default()
    };
    let all = [
        krc("krc 1", 1.0, false),
        krc("krc 0.25", 0.25, false),
        krc("krc+resample 1", 1.0, true),
        krc("krc+resample 0.25", 0.25, true),
        krcc("krc+rs cell8 1", 1.0, true, 8.0),
        krcc("krc+rs cell8 0.25", 0.25, true, 8.0),
        krcc("krc+rs cell16 1", 1.0, true, 16.0),
        krcc("krc+rs cell16 0.25", 0.25, true, 16.0),
        krcr("krc+rs+ru cell8 1", 1.0, true, 8.0, true),
        krcr("krc+rs+ru cell8 0.25", 0.25, true, 8.0, true),
        krcr("krc+rs+ru cell16 0.25", 0.25, true, 16.0, true),
        krcd("krc+rs+dm cell8 1", 1.0, true, 8.0, false, true),
        krcd("krc+rs+dm cell8 0.25", 0.25, true, 8.0, false, true),
        krcd("krc+rs+dm cell16 1", 1.0, true, 16.0, false, true),
        krcd("krc+rs+dm cell16 0.25", 0.25, true, 16.0, false, true),
        krcs("krc+rs+sd cell8 1", 1.0, true, 8.0, false, false, true),
        krcs("krc+rs+sd cell8 0.25", 0.25, true, 8.0, false, false, true),
        krcs("krc+rs+sd cell16 0.25", 0.25, true, 16.0, false, false, true),
        krcs("krc+rs+sd+dm cell8 1", 1.0, true, 8.0, false, true, true),
        krcs("krc+rs+sd+dm cell8 0.25", 0.25, true, 8.0, false, true, true),
        krcs("krc+rs+sd+dm cell16 0.25", 0.25, true, 16.0, false, true, true),
        krcs("krc+rs+sd+ru cell8 1", 1.0, true, 8.0, true, false, true),
        krcs("krc+rs+sd+ru cell16 0.25", 0.25, true, 16.0, true, false, true),
        krcs("krc+rs+sd+dm cell4 1", 1.0, true, 4.0, false, true, true),
        krcs("krc+rs+sd+dm cell6 1", 1.0, true, 6.0, false, true, true),
        krcs("krc+rs+sd+dm cell6 0.5", 0.5, true, 6.0, false, true, true),
        nv("sharc_nv 1px/5x5", 5),
        nv("sharc_nv 1px/1x1", 1),
    ];
    let mut v = all.to_vec();
    // SHaRC tuned for responsiveness (short accumulation window, a setting the sample exposes)
    for (name, down, acc) in [("sharc_nv acc5 1px/1x1", 1u32, 5u32), ("sharc_nv acc5 1px/2x2", 2, 5), ("sharc_nv acc10 1px/1x1", 1, 10)] {
        v.push(rt::DuelCfg {
            name,
            method: Method::Sharc,
            ppp: 1.0,
            resample: false,
            cell_px: 4.0,
            reuse: false,
            demod: false,
            seed: false,
            support: false,
            tau: 0.0,
            multi: false,
            sh: ShParams { downscale: down, accum_frames: acc, ..d },
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            drs: false,
        });
    }
    for (name, cell, ppp) in [("v2m cell4 0.5", 4.0f32, 0.5f32), ("v2m cell4 1", 4.0, 1.0), ("v2m cell6 0.5", 6.0, 0.5), ("v2m cell6 0.25", 6.0, 0.25)] {
        v.push(rt::DuelCfg {
            name,
            method: Method::Krc,
            ppp,
            resample: true,
            cell_px: cell,
            reuse: false,
            demod: true,
            seed: true,
            support: false,
            tau: 0.05,
            multi: true,
            sh: d,
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            drs: false,
        });
    }
    for (name, cell, ppp) in [
        ("v2 cell8 0.25", 8.0f32, 0.25f32),
        ("v2 cell8 2", 8.0, 2.0),
        ("v2 cell4 1", 4.0, 1.0),
        ("v2 cell6 0.5", 6.0, 0.5),
        ("v2 cell8 0.5", 8.0, 0.5),
        ("v2 cell8 1", 8.0, 1.0),
        ("v2 cell12 0.25", 12.0, 0.25),
    ] {
        v.push(rt::DuelCfg {
            name,
            method: Method::Krc,
            ppp,
            resample: true,
            cell_px: cell,
            reuse: false,
            demod: true,
            seed: true,
            support: false,
            tau: 0.05,
            multi: false,
            sh: d,
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            drs: false,
        });
    }
    for (name, tau) in [("krc+target 3% cell4", 0.03f32), ("krc+target 5% cell4", 0.05), ("krc+target 8% cell4", 0.08)] {
        v.push(rt::DuelCfg {
            name,
            method: Method::Krc,
            ppp: 1.0,
            resample: true,
            cell_px: 4.0,
            reuse: false,
            demod: true,
            seed: true,
            support: false,
            tau,
            multi: false,
            sh: d,
            m_light: 0,
            debt_rate: -1.0,
            reset_ratio: 0.0,
            dn_gamma: 0.0,
            drs: false,
        });
    }
    for (name, ppp, cell) in [("krc+sup cell4 1", 1.0f32, 4.0f32), ("krc+sup cell6 0.5", 0.5, 6.0), ("krc+sup cell8 0.25", 0.25, 8.0), ("krc+sup cell6 1", 1.0, 6.0)] {
        v.push(rt::DuelCfg {
            name,
            method: Method::Krc,
            ppp,
            resample: true,
            cell_px: cell,
            reuse: false,
            demod: true,
            seed: true,
            support: true,
            tau: 0.0,
        multi: false,
            sh: d,
            ..Default::default()
        });
    }
    for down in [3u32, 2] {
        v.push(rt::DuelCfg {
            name: if down == 3 { "sharc_nv 1px/3x3" } else { "sharc_nv 1px/2x2" },
            method: Method::Sharc,
            ppp: 1.0,
            resample: false,
            cell_px: 4.0,
            reuse: false,
            demod: false,
            seed: false,
            support: false,
        tau: 0.0,
        multi: false,
            sh: ShParams { downscale: down, ..d },
            ..Default::default()
        });
    }
    v
}

/// Configurations of the game pipeline: KRC v2 (per-pixel direct + indirect from the cells) and
/// official SHaRC (Render pass), with a denoiser; two raw variants for reference.
fn game_cfgs() -> Vec<(rt::DuelCfg, bool)> {
    use gpu_krc::{Method, ShParams};
    let d = ShParams::default();
    let base = rt::DuelCfg {
        name: "",
        method: Method::Krc,
        ppp: 1.0,
        resample: true,
        cell_px: 8.0,
        reuse: false,
        demod: true,
        seed: true,
        support: false,
        tau: 0.05,
        multi: false,
        sh: d,
        m_light: 0,
        debt_rate: -1.0,
        reset_ratio: 0.0,
        dn_gamma: 0.0,
        drs: false,
    };
    let krc = |name, ppp| rt::DuelCfg { name, ppp, ..base };
    let sh = |name, down, acc| rt::DuelCfg { name, method: Method::Sharc, sh: ShParams { downscale: down, accum_frames: acc, ..d }, ..base };
    let fix = |name, cell_px, m_light, debt_rate, reset_ratio, dn_gamma| rt::DuelCfg {
        name,
        ppp: 0.5,
        cell_px,
        m_light,
        debt_rate,
        reset_ratio,
        dn_gamma,
        ..base
    };
    let shg = |name, dn_gamma| rt::DuelCfg { name, dn_gamma, ..sh("", 1, 20) };
    vec![
        // fixes (KRC 0.5 path/px), one at a time then together
        // final campaign: KRC game (32 lamp correlated paths), SHaRC, clamp 1.5 and 3
        (rt::DuelCfg { drs: true, ..fix("drs krc 0.5 m32", 8.0, 32, -1.0, 0.0, 0.0) }, true),
        (rt::DuelCfg { drs: true, ..fix("drs krc 0.5 m8", 8.0, 0, -1.0, 0.0, 0.0) }, true),
        (fix("final krc 0.5 g3", 8.0, 32, -1.0, 0.0, 3.0), true),
        (rt::DuelCfg { ppp: 0.25, ..fix("final krc 0.25 g3", 8.0, 32, -1.0, 0.0, 3.0) }, true),
        (fix("final krc 0.5 g1.5", 8.0, 32, -1.0, 0.0, 0.0), true),
        (shg("final sharc 1x1 g3", 3.0), true),
        (shg("final sharc 1x1 g1.5", 0.0), true),
        (rt::DuelCfg { name: "final sharc acc5 g3", dn_gamma: 3.0, ..sh("", 1, 5) }, true),
        (rt::DuelCfg { name: "final sharc acc5 g1.5", ..sh("", 1, 5) }, true),
        (rt::DuelCfg { name: "final sharc 5x5 g3", dn_gamma: 3.0, ..sh("", 5, 20) }, true),
        (shg("fix2 sharc g3", 3.0), true),
        (fix("fix2 c16g3", 16.0, 0, -1.0, 0.0, 3.0), true),
        (fix("fix2 m32g3", 8.0, 32, -1.0, 0.0, 3.0), true),
        (fix("fix2 c16m32g3", 16.0, 32, -1.0, 0.0, 3.0), true),
        (fix("fix2 c12m32g3", 12.0, 32, -1.0, 0.0, 3.0), true),
        (fix("fix cell12", 12.0, 0, -1.0, 0.0, 0.0), true),
        (fix("fix cell16", 16.0, 0, -1.0, 0.0, 0.0), true),
        (fix("fix m32", 8.0, 32, -1.0, 0.0, 0.0), true),
        (fix("fix sansdette", 8.0, 0, 0.0, 0.0, 0.0), true),
        (fix("fix reset50", 8.0, 0, -1.0, 0.5, 0.0), true),
        (fix("fix gamma3", 8.0, 0, -1.0, 0.0, 3.0), true),
        (shg("fix sharc gamma3", 3.0), true),
        (krc("krc 0.25", 0.25), true),
        (krc("krc 0.5", 0.5), true),
        (krc("krc 1", 1.0), true),
        (sh("sharc 1px/5x5", 5, 20), true),
        (sh("sharc 1px/2x2", 2, 20), true),
        (sh("sharc 1px/1x1", 1, 20), true),
        (sh("sharc acc5 1px/1x1", 1, 5), true),
        (krc("krc 0.5", 0.5), false),
        (sh("sharc 1px/1x1", 1, 20), false),
    ]
}
