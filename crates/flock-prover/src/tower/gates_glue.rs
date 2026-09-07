// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
pub(super) use crate::r1cs_hashes::merkle_glue::{
    BitSpreadInput, BitSpreadTable, FamilyTransposeTileInput, FamilyTransposeTileTable,
    PowMaskInput, PowMaskTable, SwapInput, SwapTable,
};
use crate::tower::{F128, GateType, SLOT_WORDS, SlotWitness, TableType, digest_words, unpack8};

/// One Merkle level's conditional swap. The sibling is a [`GateType::Hint`] —
/// it is not word-aligned-wireable in the composite and nothing else reads it
/// here either, so it stays free witness.
pub(super) struct SwapGate {
    pub(super) nu: usize,
}

impl GateType for SwapGate {
    type Row = SwapInput;
    type Hint = [u32; SLOT_WORDS];

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&SwapTable::build_block_r1cs(self.nu))
            .with_io_schema(SwapTable::io_schema())
    }

    fn eval(&self, inputs: &[F128], hint: &Self::Hint, outputs: &mut Vec<F128>) -> Self::Row {
        let row = SwapInput {
            bit_word: (inputs[0].lo as u128) | ((inputs[0].hi as u128) << 64),
            prev: unpack8(inputs[1], inputs[2]),
            sib: *hint,
        };
        let (left, right) = SwapTable::outputs(&row);
        let (lw, rw) = (digest_words(&left), digest_words(&right));
        outputs.extend_from_slice(&[lw[0], lw[1], rw[0], rw[1]]);
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// Relocate each of the index word's low `depth` bits into its own word, so a
/// per-level swap row can read it at the one column its uniform relation is
/// allowed to look at.
pub(super) struct BitSpreadGate {
    pub(super) ty: BitSpreadTable,
    pub(super) nu: usize,
}

impl GateType for BitSpreadGate {
    type Row = BitSpreadInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&self.ty.build_block_r1cs(self.nu))
            .with_io_schema(self.ty.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> BitSpreadInput {
        let raw = |i: usize| (inputs[i].lo as u128) | ((inputs[i].hi as u128) << 64);
        let word = raw(0);
        let zero_mask = raw(1);
        debug_assert_eq!(inputs[2], F128::ZERO);
        let position_mask = raw(3);
        let position_prefix = raw(4);
        outputs.extend((0..self.ty.depth).map(|l| F128::new(((word >> l) & 1) as u64, 0)));
        outputs.push(F128::new(
            ((word & position_mask) ^ position_prefix) as u64,
            (((word & position_mask) ^ position_prefix) >> 64) as u64,
        ));
        BitSpreadInput {
            word,
            zero_mask,
            position_mask,
            position_prefix,
        }
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

/// The fused PoW mask row: predicate prefix + nonce width in ONE 4-word
/// row — see [`PowMaskTable`] for the layout and the repurposed-high-half
/// trick that makes it fit 512 bits.
pub(super) struct PowMaskGate {
    pub(super) nu: usize,
}

/// One wired 8x8 tile of the family-H tensor-algebra transpose.  The boolean
/// relation binds the tile selector as well as all eight source and output
/// words; the element layer only has to accumulate the resulting partial dot
/// products.
pub(super) struct FamilyTransposeTileGate {
    pub(super) nu: usize,
}

impl GateType for FamilyTransposeTileGate {
    type Row = FamilyTransposeTileInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&FamilyTransposeTileTable::build_block_r1cs(self.nu))
            .with_io_schema(FamilyTransposeTileTable::io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let rows: [F128; 8] = inputs[..8].try_into().expect("eight transpose rows");
        debug_assert_eq!(inputs[8].hi, 0, "the tile selector fits one byte");
        debug_assert_eq!(inputs[8].lo >> 8, 0, "the tile selector fits one byte");
        let row = FamilyTransposeTileInput {
            rows,
            selector: inputs[8].lo as u8,
        };
        outputs.extend_from_slice(&FamilyTransposeTileTable::outputs(&row));
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

impl GateType for PowMaskGate {
    type Row = PowMaskInput;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&PowMaskTable.build_block_r1cs(self.nu))
            .with_io_schema(PowMaskTable.io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), _outputs: &mut Vec<F128>) -> PowMaskInput {
        let w = |i: usize| (inputs[i].lo as u128) | ((inputs[i].hi as u128) << 64);
        debug_assert_eq!(inputs[3], F128::new(0, 1u64 << 63));
        PowMaskInput {
            pred: w(0),
            nonce: w(1),
            mask: w(2),
        }
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}
