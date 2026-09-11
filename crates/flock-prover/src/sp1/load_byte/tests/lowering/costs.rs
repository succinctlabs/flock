use super::*;
use std::hint::black_box;
use std::time::Instant;

const CAPACITIES: [usize; 6] = [1, 2, 4, 7, 13, 16];

#[test]
fn lowering_sizes_include_auxiliaries_holes_and_power_of_two_growth() {
    for capacity in CAPACITIES {
        let chip = LoadByteCircuit::build(capacity);
        let direct = chip.lower(LoweringMode::Direct).unwrap();
        let identity = chip.lower(LoweringMode::RequireIdentityC).unwrap();
        assert_eq!(direct.value_count(), 1 + 550 * capacity);
        assert_eq!(direct.rows().len(), 577 * capacity);
        assert_eq!(direct.layout().useful_bits(), direct.rows().len());
        assert_eq!(identity.auxiliaries().len(), 27 * capacity - 1);
        assert_eq!(identity.value_count(), 604 * capacity - 1);
        assert_eq!(identity.rows().len(), identity.value_count());
        assert_eq!(identity.layout().useful_bits(), 631 * capacity - 2);
        if capacity == 13 {
            // Appending after the whole source prefix retains 350 holes.
            assert_eq!(identity.value_count().next_power_of_two(), 8192);
            assert_eq!(identity.layout().useful_bits().next_power_of_two(), 16384);
        }
    }
}

fn median_us(mut run: impl FnMut()) -> f64 {
    run();
    let mut samples = std::array::from_fn::<_, 9, _>(|_| {
        let start = Instant::now();
        run();
        start.elapsed()
    });
    samples.sort_unstable();
    samples[4].as_secs_f64() * 1e6
}

#[test]
#[ignore = "diagnostic cost report; run alone in release mode with --nocapture"]
fn lowering_cost_profile() {
    println!(
        "capacity,mode,values,rows,useful,padded,support_bytes,action_bytes,temporaries,lower_us,walk_build_us,sparse_us,forward_us"
    );
    for capacity in CAPACITIES {
        let chip_build = median_us(|| {
            black_box(LoadByteCircuit::build(capacity));
        });
        let chip = LoadByteCircuit::build(capacity);
        let inputs = chip.honest_inputs(&vec![
            event(
                LoadByteOpcode::Lb,
                0x1_0000,
                7,
                0x8000_0000_0000_0000
            );
            capacity
        ]);
        eprintln!("capacity={capacity} source_build_us={chip_build:.2}");
        for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
            let lower_time = median_us(|| {
                black_box(chip.lower(mode).unwrap());
            });
            let lowered = chip.lower(mode).unwrap();
            let walk_build = median_us(|| {
                black_box(lowered.walk_plan().unwrap());
            });
            let plan = lowered.walk_plan().unwrap();
            let sparse_time = median_us(|| {
                black_box(matrix(&lowered));
            });
            let sparse = matrix(&lowered);
            let trace = plan.forward(&inputs, sparse.k_log).unwrap();
            assert!(sparse.satisfies(&trace.z));
            assert_eq!(
                trace.z,
                physical(&lowered, &lowered.evaluate(&inputs).unwrap(), sparse.k_log)
            );
            let forward = median_us(|| {
                black_box(plan.forward(&inputs, sparse.k_log).unwrap());
            });
            let stats = plan.stats();
            println!(
                "{capacity},{mode:?},{},{},{},{},{},{},{},{lower_time:.2},{walk_build:.2},{sparse_time:.2},{forward:.2}",
                lowered.value_count(),
                lowered.rows().len(),
                lowered.layout().useful_bits(),
                1usize << sparse.k_log,
                lowered.normalized_support_bytes(),
                stats.action_bytes,
                stats.max_live_temporaries
            );
        }
    }

    // Structural comparison only: this checker still needs an outer accept binding.
    let source = LoadByteCircuit::build(2);
    let checker_build = median_us(|| {
        black_box(source.circuit().identity_checker());
    });
    let checker = source.circuit().identity_checker();
    let raw = checker.unbound_circuit();
    let plan = raw.walk_plan().unwrap();
    println!(
        "reference_acceptance_checker capacity=2 values={} rows={} useful={} padded={} support_bytes={} action_bytes={} temporaries={} build_us={checker_build:.2}; no comparable proof without accept binding",
        raw.value_count(),
        raw.row_count(),
        plan.useful_bits(),
        plan.useful_bits().next_power_of_two(),
        raw.normalized_support_bytes(),
        plan.stats().action_bytes,
        plan.stats().max_live_temporaries
    );
}
