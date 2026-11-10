//! Native witness generation for whole DSL operations.
//!
//! [`BooleanGate`](super::BooleanGate)'s packer walks a gadget bit-sliced,
//! 64 rows at a time, and transposes every column into row words. For an
//! operation that dominates a gadget — a hash compression — a hand-written
//! generator does far better: it computes each row with native word
//! arithmetic and writes its bits in row form directly, with no walk and no
//! transpose. [`BooleanGate::with_kernel`](super::BooleanGate::with_kernel)
//! attaches such a generator to every call of one operation kind: the walk
//! skips the operation's rows, hands its argument values over row by row,
//! and reads the operation's result back from its materialized columns.
//!
//! The kernel is trusted only to produce a satisfying witness, as a native
//! evaluator is: the relation is the circuit's, so a kernel that disagreed
//! with it could only break proofs.

/// A native generator for one operation kind's rows.
///
/// The operation must own a contiguous run of [`width`](Self::width)
/// columns, each defined by a row at its own position, ending with its
/// materialized result.
pub trait OperationKernel: Send + Sync {
    /// Argument bits, in the operation's declared input order.
    fn arg_bits(&self) -> usize;

    /// Columns the operation owns, from its first.
    fn width(&self) -> usize;

    /// Eight rows at once. `args[j]` holds row `j`'s argument bits as
    /// little-endian words. For each row `j`, OR its `z`, `A·z` and `B·z`
    /// bits at positions `base..base + width()` into lane `j` of `z`, `a`
    /// and `b`, whose word `w` covers block positions `64w..64w + 64`.
    /// Those positions are zero on entry.
    fn fill8(
        &self,
        args: [&[u64]; 8],
        base: usize,
        z: &mut [[u64; 8]],
        a: &mut [[u64; 8]],
        b: &mut [[u64; 8]],
    );
}
