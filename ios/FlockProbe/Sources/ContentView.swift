import SwiftUI
import os

/// One screen: runs the probe plan on launch, shows each result line, and
/// prints the same line to stdout so `simctl launch --console` and
/// `devicectl` logs carry it. Pool sizes are process-wide, so a sweep over
/// pool sizes is several launches with different arguments:
///
///   -global N -all_core M   rayon pool sizes (0 = defaults)
///   -leaf N[,N...]          leaf sizes to prove (default 256,65536)
///   -tower L,B              also run a tower of L leaves x B blocks
///   -exit 1                 quit after the plan (for command-line runs)
struct ContentView: View {
    @State private var lines: [String] = []
    @State private var running = false

    var body: some View {
        NavigationStack {
            List(lines.indices, id: \.self) { i in
                Text(lines[i]).font(.system(.footnote, design: .monospaced))
            }
            .navigationTitle("Flock probe")
            .toolbar {
                Button(running ? "Running…" : "Run again") { run() }.disabled(running)
            }
        }
        .onAppear { run() }
    }

    func run() {
        running = true
        lines = []
        // The sequential parts of a prove run on the calling thread; give it a
        // deep stack (the default secondary-thread stack on iOS is 512 KB).
        let t = Thread { runPlan { line in
            DispatchQueue.main.async { lines.append(line) }
            print("[flock] \(line)")
            fflush(stdout)
        }
        DispatchQueue.main.async { running = false }
        }
        t.stackSize = 64 << 20
        t.start()
    }
}

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: "-\(name)"), i + 1 < a.count else { return nil }
    return a[i + 1]
}

func mib(_ bytes: UInt64) -> String { String(format: "%.1f MiB", Double(bytes) / 1048576.0) }

func runPlan(_ emit: (String) -> Void) {
    let global = UInt32(arg("global") ?? "0") ?? 0
    let allCore = UInt32(arg("all_core") ?? "0") ?? 0
    let leaves = (arg("leaf") ?? "256,65536").split(separator: ",").compactMap { UInt32($0) }
    let tower = arg("tower")?.split(separator: ",").compactMap { UInt32($0) }

    let device = UIDevice.current
    emit("\(device.model) \(device.systemName) \(device.systemVersion) | cores \(ProcessInfo.processInfo.activeProcessorCount) | RAM \(mib(ProcessInfo.processInfo.physicalMemory))")
    emit("memory budget left: \(mib(UInt64(os_proc_available_memory())))")

    let rc = flock_mobile_init_threads(global, allCore)
    emit("pools: global \(flock_mobile_thread_count()) / all-core \(flock_mobile_all_core_thread_count()) (init rc \(rc))")

    for n in leaves {
        var r = FlockMobileReport()
        let t0 = Date()
        let code = flock_mobile_leaf_demo(n, &r)
        let wall = Date().timeIntervalSince(t0)
        if code != 0 { emit("leaf \(n): FAILED rc \(code)"); continue }
        let rate = Double(n) / (r.online_wall_ms / 1000)
        emit(String(format: "leaf %d: %.0fk blake3/sec", n, rate / 1000))
        let proveRate = Double(n) / (r.online_prove_ms / 1000)
        emit(String(format: "         prove only %.0fk blake3/sec (the benchmark's metric)", proveRate / 1000))
        emit(String(format: "         online %.1f ms = walk %.1f + witgen %.1f + prove %.1f",
                    r.online_wall_ms, r.online_walk_ms, r.online_witgen_ms, r.online_prove_ms))
        emit(String(format: "         cold %.1f ms | warm whole call %.1f ms | wall %.1f s", r.prove_ms, r.prove_warm_ms, wall))
        emit(String(format: "         peak RSS %@ | budget left %@", mib(r.peak_rss_bytes), mib(UInt64(os_proc_available_memory()))))
    }
    if let t = tower, t.count == 2 {
        var r = FlockMobileReport()
        let code = flock_mobile_tower_demo(t[0], t[1], &r)
        if code != 0 { emit("tower \(t[0])x\(t[1]): FAILED rc \(code)") } else {
            emit(String(format: "tower %dx%d: prove %.1f s | discharge %.1f ms | root %.1f KiB | peak RSS %@ | budget left %@",
                        t[0], t[1], r.prove_ms / 1000, r.discharge_ms, r.root_proof_kib, mib(r.peak_rss_bytes), mib(UInt64(os_proc_available_memory()))))
        }
    }
    emit("done")
    if arg("exit") == "1" { exit(0) }
}
