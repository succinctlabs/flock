use super::*;

/// The flat instruction stream computes what the action walk computes, lane
/// for lane, and rejects the same rows in the same lanes.
#[test]
fn compact_walk_matches_the_action_walk() {
    let mut rng = crate::test_rng::Rng::new(0x51_1CED);
    for mode in [LoweringMode::Direct, LoweringMode::RequireIdentityC] {
        let circuit = batch_fixture(mode);
        let plan = circuit.walk_plan().unwrap();
        let k_log = capacity_log(plan.useful_bits()).max(7);
        let positions = plan.sliced_positions();
        assert!(positions >= plan.useful_bits() && positions % 128 == 0);
        let (mut z, mut a, mut b, mut t) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut cz, mut ca, mut cb) = (Vec::new(), Vec::new(), Vec::new());
        let mut rejected = 0;
        for _ in 0..64 {
            let inputs: Vec<u64> = (0..circuit.inputs().len())
                .map(|_| rng.next_u64())
                .collect();
            let live = rng.next_u64() | 1;
            let want =
                plan.forward_sliced_into(&inputs, live, k_log, &mut z, &mut a, &mut b, &mut t);
            let got = plan.forward_sliced_compact(&inputs, live, &mut cz, &mut ca, &mut cb);
            assert_eq!(got, want, "{mode:?}");
            if want.is_err() {
                rejected += 1;
                continue;
            }
            let n = plan.useful_bits();
            assert_eq!(cz[..n], z[..n], "{mode:?} z");
            assert_eq!(ca[..n], a[..n], "{mode:?} A·z");
            assert_eq!(cb[..n], b[..n], "{mode:?} B·z");
            assert!(cz[n..positions].iter().all(|&w| w == 0), "{mode:?} padding");
        }
        // The fixture's general constraint rejects most random lanes in
        // direct mode; identity-C lowering turns it into free cancellation
        // bits, which random inputs fill in rejected ways too.
        assert!(rejected > 0, "{mode:?}: no rejection exercised");
    }
}
