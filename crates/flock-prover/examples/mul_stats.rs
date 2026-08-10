use flock_core::r1cs::SparseBinaryMatrix;
use flock_prover::r1cs_hashes::{mul64, mul64_karatsuba as mulk};

fn nnz_of(a: &SparseBinaryMatrix, b: &SparseBinaryMatrix) -> usize {
    a.rows.iter().map(|r| r.len()).sum::<usize>() + b.rows.iter().map(|r| r.len()).sum::<usize>()
}
// Model fitted to the two measured points (ST, m=28, TR 7970X):
//   t_ns = 1.296*bits + 4834/muls_per_block + 0.001572*nnz
// schoolbook -> 11.25 (meas 11.27), karatsuba -> 8.12 (meas 8.11)
fn model(bits: usize, nnz: usize, mpb: usize) -> f64 {
    1.296 * bits as f64 + 4834.0 / mpb as f64 + 0.001572 * nnz as f64
}
fn line(name: &str, bits: usize, mpb: usize, nnz: usize, nodes: usize, ok: &str) {
    println!("{name:<28} {bits:>5} {mpb:>3} {:>5.1}% {nnz:>7} {:>5.1} {nodes:>6} {:>7.2}µs {ok}",
        100.0*(mpb*bits) as f64/(1usize<<mulk::K_LOG) as f64,
        nnz as f64/bits as f64, model(bits,nnz,mpb)/1000.0);
}
fn main() {
    println!("{:<28} {:>5} {:>3} {:>6} {:>7} {:>5} {:>6} {:>9}",
             "variant","bits","/bk","fill","nnz","nz/r","nodes","modelled");
    let (a,b)=mul64::build_matrices();
    let c0=mul64::circuit();
    line("schoolbook", mul64::SUB_BITS, mul64::MULS_PER_BLOCK,
         nnz_of(&a,&b)/mul64::MULS_PER_BLOCK, c0.nodes.len(), "[meas 11.27]");
    let c=mulk::circuit(); let (a,b)=mulk::build_matrices();
    line("default (subtractive)", c.sub_bits, c.muls_per_block,
         nnz_of(&a,&b)/c.muls_per_block, c.nodes.len(), "[meas 8.11]");
    println!();
    let mut best=(f64::MAX,String::new());
    for base in [4usize,8,16] { for bind in [0usize,1,2,3] { for bz1 in [true,false] {
        if bind==0 && !bz1 { continue }
        let (bits,nnz,mpb,ok,nodes)=mulk::stats_sub(base,bind,bz1);
        let tag=format!("sub base{base} bind{bind}{}", if bz1{""} else{" z0z2"});
        line(&tag,bits,mpb,nnz,nodes,if ok{"ok"}else{"*** FAIL ***"});
        let t=model(bits,nnz,mpb);
        if t<best.0 { best=(t,tag); }
    }}}
    println!("\nbest modelled: {} at {:.2}µs", best.1, best.0/1000.0);
}
