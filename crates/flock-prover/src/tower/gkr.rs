use super::*;
use flock_core::circuit::CellSlot;
use flock_core::field::PHI_8_TABLE;
use flock_core::zerocheck::K_SKIP;
use flock_core::zerocheck::univariate_skip::build_eq;
use flock_hash::blake3_compress;

/// One wiring-GKR layer, located on the tape (the assembly's wire map).
pub(super) struct GkrLayerRec {
    pub(super) lam_fin: usize,
    pub(super) rounds: Vec<(usize, usize)>, // (g_v, squeeze fin)
    pub(super) g0s: Vec<F128>,
    pub(super) v_v: usize, // vl0; vl1/vr0/vr1 follow
    pub(super) ck_fin: usize,
}
pub(super) struct GkrRec {
    pub(super) alpha_fin: usize,
    pub(super) beta_fin: usize,
    pub(super) top_v: usize,
    pub(super) layers: Vec<GkrLayerRec>,
    pub(super) fgs_v: usize, // f_eval, g_eval, s_sigma consecutive
    pub(super) r_pt: Vec<F128>,
}

/// **THE RECOMBINATION (round 4), pinned natively.** The wiring verifier's
/// `ŵ(ρ) = Σ_gate eq_slot[ι]·gather[ι] + Σ_public eq_slot[ι]·⟨eq_row, slot⟩`
/// (`circuit.rs` verify_wiring_core) is the ONE check that reads the child's
/// publics — and, with `f_eval == g_eval` beside it, was enforced only by the
/// tape constructors' scaffolding-tier native verify, never by the parent's
/// statement. This replica recomputes both from LOCATED tape words — the
/// gather pd values, the child's public segment, the GKR squeeze point — so
/// the emission has a pinned reference for every wire it binds. Also pins the
/// pd-claim order the emitter indexes: `[element c, element lc, gathers in
/// cell-slot enumeration order]`.
///
/// Returns `num_public_slots` (the emitters derive everything else from the
/// gather count and `n_log_i`; `cells.nu()` is asserted against it here).
pub(super) fn pin_recombination(
    cells: &flock_core::circuit::CellSpace,
    n_log_i: usize,
    public: &[F128],
    gather: &[F128],
    gammas: &[PdRec],
    n_el_pd: usize,
    vals_rec: &[F128],
    r_pt: &[F128],
    fgs_v: usize,
) -> usize {
    let (nu_c, c_bits) = (cells.nu(), cells.c_bits());
    assert_eq!(nu_c, n_log_i, "the cell space's row vars are the union's");
    assert_eq!(r_pt.len(), nu_c + c_bits, "ρ spans the cell space");
    assert_eq!(
        gather.len(),
        cells.num_gate_slots(),
        "one gather per gate slot"
    );
    assert_eq!(
        gammas.len(),
        n_el_pd + gather.len(),
        "pd claims = the element (c, lc) pair, when the class exists, + the gathers"
    );
    for (i, g) in gather.iter().enumerate() {
        assert_eq!(
            vals_rec[gammas[n_el_pd + i].val_v],
            *g,
            "gather {i} is pd claim {} on the stream",
            n_el_pd + i
        );
    }
    let eq_row = build_eq(&r_pt[..nu_c]);
    let eq_slot = build_eq(&r_pt[nu_c..]);
    let mut acc = F128::ZERO;
    for (iota, slot) in cells.slots().iter().enumerate() {
        match *slot {
            CellSlot::Gate { .. } => acc += eq_slot[iota] * gather[iota],
            CellSlot::Public { s } => {
                let base = s << nu_c;
                let hi = ((base + (1usize << nu_c)).min(public.len())).saturating_sub(base);
                let mut v = F128::ZERO;
                for j in 0..hi {
                    v += eq_row[j] * public[base + j];
                }
                acc += eq_slot[iota] * v;
            }
            CellSlot::Pad => {}
        }
    }
    assert_eq!(
        acc, vals_rec[fgs_v],
        "the gathers + publics-MLE recombine to the absorbed f_eval"
    );
    assert_eq!(
        vals_rec[fgs_v],
        vals_rec[fgs_v + 1],
        "f_eval == g_eval on the stream"
    );
    cells.num_public_slots()
}

/// The eq table's `live` prefix over `point_w` wires (LSB-first —
/// `build_eq`'s convention), as MacGate rows: the DOUBLING build, one row
/// per node — `e·ρ` is `0 + e·ρ` and `e·(1+ρ)` is `e + e·ρ`, so both
/// children of a node are single MAC rows. Rows, not advice: every weight is
/// wire-bound to its squeeze. Ancestors of the live prefix are themselves a
/// prefix (low bits), so level `i` builds `min(2^i, live)` entries.
pub(super) fn emit_eq_prefix(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    point_w: &[Wire],
    live: usize,
    zw: Wire,
    ow: Wire,
) -> Vec<Wire> {
    let live = live.max(1);
    let mut eq_w: Vec<Wire> = vec![ow];
    for (i, &rw) in point_w.iter().enumerate() {
        let half = 1usize << i;
        let width = (2 * half).min(live);
        let mut next = Vec::with_capacity(width);
        for x in 0..width.min(half) {
            next.push(sb.gate(macs, &[eq_w[x], eq_w[x], rw])[0]);
        }
        for x in half..width {
            next.push(sb.gate(macs, &[zw, eq_w[x - half], rw])[0]);
        }
        eq_w = next;
    }
    eq_w
}

/// Wire identities for every claim emitted by `SigmaAssertion::claims`, in
/// exactly the same order. The accumulator's circuit-structure table binds
/// Product-GKR's masked-ID/live/sigma evaluations, Boolean count-prefix
/// values, and element affine-constant strips under one child digest.
#[allow(clippy::too_many_arguments)]
pub(super) fn circuit_structure_claim_wires(
    sigma: &flock_core::circuit::SigmaAssertion,
    gkr_point: &[Wire],
    masked_id_w: Wire,
    live_w: Wire,
    sigma_w: Wire,
    boolean_point: &[Wire],
    boolean_values: &[(usize, Wire)],
    element_point: Option<&[Wire]>,
    element_values: Option<(Wire, Wire)>,
    zw: Wire,
    ow: Wire,
) -> Vec<(Vec<Wire>, Vec<Wire>, Wire)> {
    let bit = |b: usize| if b == 0 { zw } else { ow };
    let selector = |plane: usize| -> [Wire; 3] {
        [bit(plane & 1), bit((plane >> 1) & 1), bit((plane >> 2) & 1)]
    };
    let mut base_point = gkr_point[sigma.nu..].to_vec();
    base_point.resize(sigma.base_bits, zw);
    let mut out = Vec::new();
    for (plane, value_w) in [(0, masked_id_w), (1, live_w), (2, sigma_w)] {
        let mut col = base_point.clone();
        col.extend_from_slice(&selector(plane));
        out.push((gkr_point[..sigma.nu].to_vec(), col, value_w));
    }
    assert_eq!(
        boolean_values.len(),
        sigma.boolean_pins.len(),
        "Boolean pin wires"
    );
    for ((type_index, point, _), (wire_type, value_w)) in sigma
        .boolean_pins
        .iter()
        .zip(boolean_values.iter().copied())
    {
        assert_eq!(*type_index, wire_type, "Boolean pin slot order");
        assert_eq!(point.len(), boolean_point.len(), "Boolean pin point width");
        let mut col: Vec<Wire> = (0..sigma.base_bits)
            .map(|j| bit((type_index >> j) & 1))
            .collect();
        col.extend_from_slice(&selector(5));
        out.push((boolean_point.to_vec(), col, value_w));
    }
    if let Some((point, _, _)) = &sigma.element_constants {
        let point_w = element_point.expect("element structure point wires");
        assert_eq!(point.len(), point_w.len(), "element constant point width");
        let (a_w, b_w) = element_values.expect("element constant value wires");
        for (plane, value_w) in [(3, a_w), (4, b_w)] {
            let mut col = point_w.to_vec();
            col.resize(sigma.base_bits, zw);
            col.extend_from_slice(&selector(plane));
            out.push((vec![zw; sigma.nu], col, value_w));
        }
    }
    assert_eq!(
        out.len(),
        sigma.claims().len(),
        "structure claim wire count"
    );
    out
}

/// `seed * product_j eq(left[j], right[j])`, chunked through the shared
/// prefix-product gate.  `PrefixGate` uses the characteristic-two identity
/// `eq(a,b) = 1 + a + b` for Boolean `b`; every right-hand coordinate used
/// here is either a fixed prefix bit or a Fiat--Shamir wire constrained by
/// the transcript circuit.
pub(super) fn emit_eq_product(
    sb: &mut ShapeBuilder,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    seed: Wire,
    left: &[Wire],
    right: &[Wire],
    zw: Wire,
    ow: Wire,
) -> Wire {
    assert_eq!(left.len(), right.len(), "eq-product arity");
    let mut acc = seed;
    for (aa, bb) in left.chunks(pf_w).zip(right.chunks(pf_w)) {
        let mut inputs = vec![acc];
        inputs.extend_from_slice(aa);
        inputs.extend(std::iter::repeat_n(zw, pf_w - aa.len()));
        inputs.extend_from_slice(bb);
        inputs.extend(std::iter::repeat_n(zw, pf_w - bb.len()));
        inputs.push(ow);
        acc = sb.gate(pfslot, &inputs)[0];
    }
    acc
}

pub(super) fn prefix_bit_wires(bits: usize, n: usize, zw: Wire, ow: Wire) -> Vec<Wire> {
    (0..n)
        .map(|j| if (bits >> j) & 1 == 0 { zw } else { ow })
        .collect()
}

/// The fourth output of `SpineGate` is the same `acc + x*y` primitive as
/// `MacGate`.  Assertion checks use this existing, lower-occupancy slot so
/// they do not push the recursion envelope's main MAC slot over `2^nu`.
pub(super) fn assertion_mac(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    acc: Wire,
    x: Wire,
    y: Wire,
    zw: Wire,
) -> Wire {
    sb.gate(spine, &[zw, zw, zw, acc, zw, zw, x, y, zw])[3]
}

/// Enforce the scalar-only half of `MatrixAssertion::check_reported`.
/// The matrix evaluations themselves are fold claims and eventually
/// discharge against the digest-keyed matrices; this relation binds those
/// values to the transcript-derived lincheck target inside the recursive
/// circuit.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_boolean_reported_check(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    registry: &crate::schedule::Registry,
    alpha_w: Wire,
    x_inner_w: &[Wire],
    rr_w: &[Wire],
    z_partial_w: &[Wire],
    beta_w: &[Option<Wire>],
    eval_w: &[(Wire, Wire)],
    target_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    assert_eq!(eval_w.len(), registry.num_boolean(), "Boolean eval count");
    assert_eq!(beta_w.len(), registry.num_boolean(), "Boolean beta count");
    assert_eq!(z_partial_w.len(), 1usize << K_SKIP, "Boolean low weight");
    let mut acc = zw;
    for (t, ((ty, layout), &(va_w, vb_w))) in registry
        .boolean_types()
        .iter()
        .zip(registry.slots())
        .zip(eval_w)
        .enumerate()
    {
        let inner = ty.k_log - K_SKIP;
        assert!(inner <= x_inner_w.len() && inner <= rr_w.len());
        let row_prefix = prefix_bit_wires(layout.prefix, x_inner_w.len() - inner, zw, ow);
        let col_prefix = prefix_bit_wires(layout.prefix, rr_w.len() - inner, zw, ow);
        let w_t = emit_eq_product(
            sb,
            pfslot,
            pf_w,
            ow,
            &x_inner_w[inner..],
            &row_prefix,
            zw,
            ow,
        );
        let p_t = emit_eq_product(sb, pfslot, pf_w, ow, &rr_w[inner..], &col_prefix, zw, ow);
        let ab = assertion_mac(sb, spine, vb_w, alpha_w, va_w, zw);
        let wp = assertion_mac(sb, spine, zw, w_t, p_t, zw);
        let term = assertion_mac(sb, spine, zw, wp, ab, zw);
        acc = assertion_mac(sb, spine, acc, term, ow, zw);

        match (ty.const_pin, beta_w[t]) {
            (Some(col), Some(beta)) => {
                let high = col >> K_SKIP;
                let high_bits = prefix_bit_wires(high, inner, zw, ow);
                let w_col = emit_eq_product(
                    sb,
                    pfslot,
                    pf_w,
                    z_partial_w[col & ((1usize << K_SKIP) - 1)],
                    &rr_w[..inner],
                    &high_bits,
                    zw,
                    ow,
                );
                let pin = assertion_mac(sb, spine, zw, p_t, beta, zw);
                acc = assertion_mac(sb, spine, acc, pin, w_col, zw);
            }
            (None, None) => {}
            _ => panic!("const-pin challenge schedule does not match the registry"),
        }
    }
    sb.connect(acc, target_w);
}

/// Enforce the scalar-only half of `ElementAssertion::check_reported`.
/// As on the Boolean side, the per-slot A/B values are separately folded
/// against static matrices; this equation binds their weighted combination
/// to the transcript-derived element-lincheck target.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_element_reported_check(
    sb: &mut ShapeBuilder,
    spine: flock_core::circuit::builder::SlotId,
    pfslot: flock_core::circuit::builder::SlotId,
    pf_w: usize,
    union: &UnionInstance<'_>,
    alpha_w: Wire,
    r_con_w: &[Wire],
    r_col_w: &[Wire],
    z_eval_w: Wire,
    eval_w: &[(Wire, Wire)],
    target_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    let layouts = union.element_slot_layout();
    assert_eq!(eval_w.len(), layouts.len(), "element eval count");
    let nu = union.n_log();
    let mut acc = zw;
    for (layout, &(va_w, vb_w)) in layouts.iter().zip(eval_w) {
        let kappa = layout.kappa;
        assert!(kappa <= r_con_w.len() && kappa <= r_col_w.len());
        let bits = prefix_bit_wires(layout.region_prefix(nu), r_con_w.len() - kappa, zw, ow);
        let w_r = emit_eq_product(sb, pfslot, pf_w, ow, &r_con_w[kappa..], &bits, zw, ow);
        let w_col = emit_eq_product(sb, pfslot, pf_w, ow, &r_col_w[kappa..], &bits, zw, ow);
        let ab = assertion_mac(sb, spine, va_w, alpha_w, vb_w, zw);
        let wp = assertion_mac(sb, spine, zw, w_r, w_col, zw);
        let term = assertion_mac(sb, spine, zw, wp, ab, zw);
        acc = assertion_mac(sb, spine, acc, term, ow, zw);
    }
    let rhs = assertion_mac(sb, spine, zw, acc, z_eval_w, zw);
    sb.connect(rhs, target_w);
}

/// PHASE D (docs/ag-recursion-plan.md): the AG z_skip point's IN-CIRCUIT
/// binding — the Tier-1 successor of the Tier-0 decode checker. From the
/// child's transcript wires (the two seed squeezes and the absorbed 4-byte
/// nonce), two BLAKE3 rows recompute the nonce-seed `ns = H(seed ‖ nonce)`
/// and its first XOF block; a PowMask row enforces the fused grinding
/// target on `ns[16..32]` (the Phase-A convention alignment cashing in);
/// and the published point coordinates are constrained to a fiber point
/// over the XOF-derived `x`:
///
///   x = XOF word 0                       (a wire connect)
///   t² + t = x³ + x                      (advice t — the factored base
///   y² + u·y = x·u·t,   u = x + 1        fiber with s eliminated: y = u·s
///                                        turns both AS levels inverse-free)
///   D₀·(z₁² + z₁) = Σⱼ P₀ⱼ(x)·yʲ         (denominators cleared; D₀, D₁
///   D₀D₁·(z₂² + z₂) = D₁·Σⱼ<₃ P₁ⱼ·yʲ     guarded nonzero by advice
///                       + D₀·P₁₃·y³       inverses against a fixed ONE)
///   D₀·(z₃² + z₃) = Σⱼ P₂ⱼ(x)·yʲ
///
/// CANONICITY IS RELAXED BY DESIGN: any of the ≤ 32 fiber points over `x`
/// satisfies these rows — the sampler's slot/choice bits are deliberately
/// unbound, and the fiber's 5 bits of prover freedom are repaid by the
/// all-explicit `R1_POW_BITS = 9` fused target (the schedule constant
/// moved with this emitter; the total stays `bits_for(474)`). The checker
/// items left on this surface are the nonce range and `lows == bf(point)`
/// ([`check_ag_skip_publics`] — the PowMask row pins only the nonce
/// word's high half, and Chain100 emits no row).
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_ag_point_binding(
    sb: &mut ShapeBuilder,
    b3: flock_core::circuit::builder::SlotId,
    pow: flock_core::circuit::builder::SlotId,
    macs: flock_core::circuit::builder::SlotId,
    iv: [Wire; 2],
    seed_w: [Wire; 2],
    nonce_w: Wire,
    pt_w: &[Wire; 5],
    seed_n: [F128; 2],
    nonce_n: u32,
    pt_n: &flock_core::genus95_curve_code::EvaluationPoint,
    ag_r1_bits: Option<u32>,
    vals: &mut Vec<F128>,
    consts: &mut Vec<(F128, Wire)>,
    zw: Wire,
    ow: Wire,
) {
    let flags = CHUNK_START | CHUNK_END | ROOT;
    // ---- native replicas of the decode chain the rows must reproduce ----
    let seed_bytes = ag_seed_bytes(seed_n[0], seed_n[1]);
    let mut block36 = [0u8; 64];
    block36[..32].copy_from_slice(&seed_bytes);
    block36[32..36].copy_from_slice(&nonce_n.to_le_bytes());
    let words36: [u32; 16] =
        std::array::from_fn(|i| u32::from_le_bytes(block36[4 * i..4 * i + 4].try_into().unwrap()));
    let ns16 = blake3_compress(&IV, &words36, 0, 36, flags);
    let ns_bytes: [u8; 32] = std::array::from_fn(|i| (ns16[i / 4] >> (8 * (i % 4))) as u8);
    let mut block32 = [0u8; 64];
    block32[..32].copy_from_slice(&ns_bytes);
    let words32: [u32; 16] =
        std::array::from_fn(|i| u32::from_le_bytes(block32[4 * i..4 * i + 4].try_into().unwrap()));
    let xof16 = blake3_compress(&IV, &words32, 0, 32, flags);
    let x_native = F128::new(
        u64::from(xof16[0]) | (u64::from(xof16[1]) << 32),
        u64::from(xof16[2]) | (u64::from(xof16[3]) << 32),
    );
    assert_eq!(x_native, pt_n.x, "the XOF x is the point's x");
    if let Some(bits) = ag_r1_bits {
        assert!(
            flock_core::challenger::has_leading_zero_bits(&ns_bytes[16..32], bits),
            "the honest nonce clears the fused target"
        );
    }
    let (x, y) = (pt_n.x, pt_n.y);
    let u_n = x + F128::ONE;
    let t_n = if x == F128::ZERO || u_n == F128::ZERO {
        F128::ZERO
    } else {
        let s = y * u_n.inv();
        u_n * (s * s + s) * x.inv()
    };
    assert_eq!(t_n * t_n + t_n, x * x * x + x, "t solves the base AS level");
    assert_eq!(y * y + u_n * y, x * u_n * t_n, "y sits on the fiber over x");
    let mut xp = [F128::ONE; 12];
    for d in 1..12 {
        xp[d] = xp[d - 1] * x;
    }
    let d0_n = xp[10] + xp[4] + F128::ONE;
    let d1_n = xp[11] + xp[10] + xp[5] + xp[4] + xp[1] + F128::ONE;
    let (d0_inv_n, d1_inv_n) = (
        d0_n.inv(), // the honest decode rejected D₀ = 0 nonces
        d1_n.inv(),
    );
    // Native replicas of the cleared AS equations (mirroring the sampler's
    // rhs masks) — the method-note discipline before the rows land.
    let pv = |degs: &[usize], plus_one: bool| -> F128 {
        degs.iter()
            .fold(if plus_one { F128::ONE } else { F128::ZERO }, |acc, &d| {
                acc + xp[d]
            })
    };
    let yp_n = [F128::ONE, y, y * y, y * y * y];
    let z_n = [pt_n.z1, pt_n.z2, pt_n.z3];
    for (i, (zi, p)) in [
        (
            z_n[0],
            [
                pv(&[9, 6, 5, 4, 3, 2], false),
                pv(&[5, 4, 3, 2], true),
                pv(&[4, 3, 2], false),
                pv(&[3], true),
            ],
        ),
        (
            z_n[2],
            [
                pv(&[6, 4, 3, 2], false),
                pv(&[5], true),
                pv(&[3, 2, 1], true),
                pv(&[3, 2], false),
            ],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let rhs: F128 = (0..4).fold(F128::ZERO, |acc, j| acc + p[j] * yp_n[j]);
        assert_eq!(
            d0_n * (zi * zi + zi),
            rhs,
            "cleared AS level {} closes at the honest point",
            [1, 3][i]
        );
    }
    {
        let p1_n = [
            pv(&[8, 5, 4, 3, 2, 1], false),
            pv(&[6, 5, 2, 1], false),
            pv(&[3, 1], false),
            pv(&[4, 2, 1], false),
        ];
        let inner: F128 = (0..3).fold(F128::ZERO, |acc, j| acc + p1_n[j] * yp_n[j]);
        assert_eq!(
            d0_n * d1_n * (z_n[1] * z_n[1] + z_n[1]),
            d1_n * inner + d0_n * p1_n[3] * yp_n[3],
            "cleared AS level 2 closes at the honest point"
        );
    }

    // ---- the two BLAKE3 rows + the fused-PoW row ----
    let params36 = cw(sb, vals, consts, pack_params(0, 36, flags));
    let ns_out = sb.gate(
        b3,
        &[iv[0], iv[1], seed_w[0], seed_w[1], nonce_w, zw, params36],
    );
    let params32 = cw(sb, vals, consts, pack_params(0, 32, flags));
    let xof_out = sb.gate(b3, &[iv[0], iv[1], ns_out[0], ns_out[1], zw, zw, params32]);
    sb.connect(xof_out[0], pt_w[0]);
    if let Some(bits) = ag_r1_bits {
        emit_pow_checks(
            sb,
            b3,
            pow,
            iv,
            &[([ns_out[1], nonce_w], bits)],
            vals,
            consts,
        );
    }

    // ---- the fiber algebra over the published point wires ----
    let zass = cw(sb, vals, consts, F128::ZERO);
    let one_w = cw(sb, vals, consts, F128::ONE);
    let (xw, yw) = (pt_w[0], pt_w[1]);
    let zwires = [pt_w[2], pt_w[3], pt_w[4]];
    // x-powers x²..x^11 (x⁰/x¹ are ow/xw).
    let mut xpw = [ow; 12];
    xpw[1] = xw;
    for d in 2..12 {
        xpw[d] = sb.gate(macs, &[zw, xpw[d - 1], xw])[0];
    }
    // y-powers y², y³.
    let y2w = sb.gate(macs, &[zw, yw, yw])[0];
    let y3w = sb.gate(macs, &[zw, y2w, yw])[0];
    let ypw = [ow, yw, y2w, y3w];
    // C1: t² + t + x³ + x == 0.
    vals.push(t_n);
    let tw = sb.input();
    let t2w = sb.gate(macs, &[zw, tw, tw])[0];
    let mut c1 = sb.gate(macs, &[t2w, tw, ow])[0];
    c1 = sb.gate(macs, &[c1, xpw[3], ow])[0];
    c1 = sb.gate(macs, &[c1, xw, ow])[0];
    sb.connect(c1, zass);
    // C2: y² + u·y + x·u·t == 0 with u = x + 1.
    let uw = sb.gate(macs, &[ow, xw, ow])[0];
    let xuw = sb.gate(macs, &[zw, xw, uw])[0];
    let xutw = sb.gate(macs, &[zw, xuw, tw])[0];
    let mut c2 = sb.gate(macs, &[y2w, uw, yw])[0];
    c2 = sb.gate(macs, &[c2, xutw, ow])[0];
    sb.connect(c2, zass);
    // D₀, D₁ + their nonzero guards (advice inverses against the fixed ONE).
    let mut d0w = sb.gate(macs, &[ow, xpw[4], ow])[0];
    d0w = sb.gate(macs, &[d0w, xpw[10], ow])[0];
    let mut d1w = sb.gate(macs, &[ow, xw, ow])[0];
    for d in [4, 5, 10, 11] {
        d1w = sb.gate(macs, &[d1w, xpw[d], ow])[0];
    }
    vals.push(d0_inv_n);
    let d0iw = sb.input();
    let g0 = sb.gate(macs, &[zw, d0w, d0iw])[0];
    sb.connect(g0, one_w);
    vals.push(d1_inv_n);
    let d1iw = sb.input();
    let g1 = sb.gate(macs, &[zw, d1w, d1iw])[0];
    sb.connect(g1, one_w);
    // The three Artin–Schreier levels, denominators cleared. The Pᵢⱼ masks
    // mirror `sampling::sample_artin_schreier_rhs_coeffs_cached` exactly
    // (there `d0`/`d1` NAME the inverses; clearing multiplies them away).
    let poly = |sb: &mut ShapeBuilder, degs: &[usize], plus_one: bool| -> Wire {
        let mut acc = if plus_one { ow } else { zw };
        for &d in degs {
            acc = sb.gate(macs, &[acc, xpw[d], ow])[0];
        }
        acc
    };
    let p0: [Wire; 4] = [
        poly(sb, &[9, 6, 5, 4, 3, 2], false),
        poly(sb, &[5, 4, 3, 2], true),
        poly(sb, &[4, 3, 2], false),
        poly(sb, &[3], true),
    ];
    let p1: [Wire; 4] = [
        poly(sb, &[8, 5, 4, 3, 2, 1], false),
        poly(sb, &[6, 5, 2, 1], false),
        poly(sb, &[3, 1], false),
        poly(sb, &[4, 2, 1], false),
    ];
    let p2: [Wire; 4] = [
        poly(sb, &[6, 4, 3, 2], false),
        poly(sb, &[5], true),
        poly(sb, &[3, 2, 1], true),
        poly(sb, &[3, 2], false),
    ];
    // i = 0, 2 (all over D₀): D₀·(z² + z) + Σⱼ Pⱼ·yʲ == 0.
    for (zi, p) in [(zwires[0], &p0), (zwires[2], &p2)] {
        let z2 = sb.gate(macs, &[zw, zi, zi])[0];
        let z2z = sb.gate(macs, &[z2, zi, ow])[0];
        let mut acc = sb.gate(macs, &[zw, d0w, z2z])[0];
        for j in 0..4 {
            acc = sb.gate(macs, &[acc, p[j], ypw[j]])[0];
        }
        sb.connect(acc, zass);
    }
    // i = 1 (mixed D₀/D₁): D₀D₁·(z²+z) + D₁·Σⱼ<₃ Pⱼ·yʲ + D₀·P₃·y³ == 0.
    {
        let zi = zwires[1];
        let z2 = sb.gate(macs, &[zw, zi, zi])[0];
        let z2z = sb.gate(macs, &[z2, zi, ow])[0];
        let d0d1 = sb.gate(macs, &[zw, d0w, d1w])[0];
        let mut inner = sb.gate(macs, &[zw, p1[0], ow])[0];
        inner = sb.gate(macs, &[inner, p1[1], yw])[0];
        inner = sb.gate(macs, &[inner, p1[2], y2w])[0];
        let p3y3 = sb.gate(macs, &[zw, p1[3], y3w])[0];
        let mut acc = sb.gate(macs, &[zw, d0d1, z2z])[0];
        acc = sb.gate(macs, &[acc, d1w, inner])[0];
        acc = sb.gate(macs, &[acc, d0w, p3y3])[0];
        sb.connect(acc, zass);
    }
}

/// **THE LAGRANGE ROW LOWS in-circuit (round 4).** The 64 weights
/// `L_i(z_skip) = Z_N(z)·(z + λ_i)^{-1}·den^{-1}` a merge fold's boolean
/// claims carry, derived from the child's z_skip WIRE instead of published
/// and checker-rebuilt: `t_i = z + λ_i` against the shared λ const wires,
/// `Z = Π t_i` (a MAC chain — no subspace recursion needed, the factors are
/// already wires), the inverses as ADVICE bound by `t_i·y_i = 1` rows
/// (witness, not publics), and `w_i = (Z·den^{-1})·y_i`. The caller connects
/// each `w_i` to the fold's absorbed low word.
///
/// `z` on a node has no inverse witness — the ≈2^-121 completeness caveat a
/// fixed-topology circuit carries in its soundness accounting instead of a
/// branch (`lagrange_weights_on_coset`'s own posture, same constant).
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_lagrange_lows(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    lam_w: &[Wire],
    deninv_w: Wire,
    zskip_w: Wire,
    z_native: F128,
    vals: &mut Vec<F128>,
    zw: Wire,
    ow: Wire,
    zassert: Wire,
) -> Vec<Wire> {
    let ell = lam_w.len();
    let mut t_w = Vec::with_capacity(ell);
    let mut z_acc = ow;
    for &lw2 in lam_w {
        let t = sb.gate(macs, &[zskip_w, lw2, ow])[0];
        z_acc = sb.gate(macs, &[zw, z_acc, t])[0];
        t_w.push(t);
    }
    let scale = sb.gate(macs, &[zw, z_acc, deninv_w])[0];
    (0..ell)
        .map(|i| {
            let ti = z_native + PHI_8_TABLE[i];
            assert!(!ti.is_zero(), "z_skip on a φ8 node (≈2^-121)");
            vals.push(ti.inv());
            let y = sb.input();
            // 1 + t·y == 0 (char 2), into the dedicated assert-zero anchor —
            // connecting a producer into the ubiquitous `ow` class is the
            // recorded Cyclic trap.
            let delta = sb.gate(macs, &[ow, t_w[i], y])[0];
            sb.connect(delta, zassert);
            sb.gate(macs, &[zw, scale, y])[0]
        })
        .collect()
}

/// **THE RECOMBINATION in-circuit (round 4).** Rebuild `ŵ(ρ)` from the
/// absorbed gather pd wires and the H region's publics wires, CONNECT it to
/// the absorbed `f_eval`, and connect `f == g` — the two `verify_wiring_core`
/// checks that until now rode only the tape constructors' scaffolding
/// verify. The publics half is the recorded design: the H region's wires
/// feed 8-lane LeafEval folds at `ρ_row[..3]` (the "leaf arithmetic joins
/// the openings" pattern) with hi-group eq weights from the doubling build;
/// the gate half is an eq_slot-weighted MAC chain over the gather wires.
/// Zero new publics, inputs, or slot types — the checker walks are
/// untouched. Dataflow is acyclic: ρ wires come from chain rows BEFORE the
/// `(f, g, s_σ)` absorb, and `f_w` feeds only LATER chain rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_recombination(
    sb: &mut ShapeBuilder,
    macs: flock_core::circuit::builder::SlotId,
    le8: flock_core::circuit::builder::SlotId,
    pub_w: &[Wire],
    gather_w: &[Wire],
    pt_w: &[Wire],
    nu_c: usize,
    n_pub_slots: usize,
    f_w: Wire,
    g_w: Wire,
    zw: Wire,
    ow: Wire,
) {
    sb.connect(f_w, g_w);
    let rows = 1usize << nu_c;
    assert_eq!(
        pub_w.chunks(rows).count(),
        n_pub_slots,
        "public slots tile the child's segment"
    );
    let eq_slot_w = emit_eq_prefix(
        sb,
        macs,
        &pt_w[nu_c..],
        gather_w.len() + n_pub_slots,
        zw,
        ow,
    );
    let max_chunks = pub_w
        .chunks(rows)
        .map(|s| s.len().div_ceil(8))
        .max()
        .expect("a circuit child has publics");
    let eq_hi_w = emit_eq_prefix(sb, macs, &pt_w[3..nu_c], max_chunks, zw, ow);
    let mut acc = zw;
    for (i, &gw2) in gather_w.iter().enumerate() {
        acc = sb.gate(macs, &[acc, eq_slot_w[i], gw2])[0];
    }
    for (s, spub) in pub_w.chunks(rows).enumerate() {
        let mut v = zw;
        for (h, chunk) in spub.chunks(8).enumerate() {
            let mut a_in: Vec<Wire> = chunk.to_vec();
            a_in.resize(8, zw);
            a_in.extend_from_slice(&pt_w[..3]);
            a_in.push(eq_hi_w[h]);
            a_in.push(v);
            v = sb.gate(le8, &a_in)[0];
        }
        acc = sb.gate(macs, &[acc, eq_slot_w[gather_w.len() + s], v])[0];
    }
    sb.connect(acc, f_w);
}

/// The ELEMENT PIOP region, located. Round tuples are `(g_v, fin, ch)`.
pub(super) struct ElPiopRec {
    pub(super) tau_fin: usize,
    pub(super) tau_ch: usize,
    pub(super) zc_rounds: Vec<(usize, usize, usize)>,
    pub(super) eab_v: usize,
    pub(super) alpha_fin: usize,
    pub(super) alpha_ch: usize,
    pub(super) lc_rounds: Vec<(usize, usize, usize)>,
}
