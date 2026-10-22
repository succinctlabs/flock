//! Offline generator for the fixture in flock-prover's load-byte tests.
//!
//! Use `generate-sp1-load-byte-fixture.sh`; this file is copied temporarily
//! into the pinned SP1 checkout and is not part of the Flock build.

use std::borrow::Borrow;

use slop_algebra::PrimeField32;
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{MemInstrEvent, MemoryReadRecord, MemoryRecordEnum, MemoryWriteRecord},
    ExecutionRecord, ITypeRecord, Opcode,
};
use sp1_core_machine::{
    memory::load::load_byte::{LoadByteChip, LoadByteColumns},
    SupervisorMode,
};
use sp1_hypercube::air::MachineAir;
use sp1_primitives::SP1Field;

fn read(value: u64, timestamp: u64) -> MemoryRecordEnum {
    MemoryRecordEnum::Read(MemoryReadRecord {
        value,
        timestamp,
        prev_timestamp: timestamp - 1,
        prev_page_prot_record: None,
    })
}

fn write(value: u64, timestamp: u64) -> MemoryRecordEnum {
    MemoryRecordEnum::Write(MemoryWriteRecord {
        prev_timestamp: timestamp - 1,
        prev_page_prot_record: None,
        prev_value: 0,
        timestamp,
        value,
    })
}

fn main() {
    let cases: [(Opcode, u64, u64, u64); 3] = [
        (Opcode::LB, 0x1_0000, 2, 0x8877_6655_4433_2211),
        (Opcode::LB, 0x2_0000, 7, 0x8070_6050_4030_2010),
        (Opcode::LBU, 0x3_0004, 1, 0xfedc_ba98_7654_3210),
    ];
    let mut record = ExecutionRecord::default();

    for (index, (opcode, b, c, memory_value)) in cases.into_iter().enumerate() {
        let address = b.wrapping_add(c);
        let byte = (memory_value >> (8 * (address & 7))) as u8;
        let result = if opcode == Opcode::LB {
            (byte as i8 as i64) as u64
        } else {
            u64::from(byte)
        };
        let timestamp = 10 + index as u64 * 8;
        let event = MemInstrEvent::new(
            timestamp,
            0x1000 + index as u64 * 4,
            opcode,
            result,
            b,
            c,
            false,
            read(memory_value, timestamp + 1),
        );
        let adapter = ITypeRecord {
            op_a: 1,
            a: write(result, timestamp + 4),
            op_b: 2,
            b: read(b, timestamp + 3),
            op_c: c,
            is_untrusted: false,
        };
        record.memory_load_byte_events.push((event, adapter));
    }

    let chip = LoadByteChip::<SupervisorMode>::default();
    let trace: RowMajorMatrix<SP1Field> =
        chip.generate_trace(&record, &mut ExecutionRecord::default());

    for row_index in 0..cases.len() {
        let row = trace.row_slice(row_index);
        let cols: &LoadByteColumns<SP1Field, SupervisorMode> = (*row).borrow();
        let canonical = |value: SP1Field| value.as_canonical_u32();
        println!(
            "row={row_index} is_lb={} is_lbu={} address_limbs={:?} offset={:?} \
             memory_limbs={:?} selected_limb={} selected_low={} selected_byte={} msb={}",
            canonical(cols.is_lb),
            canonical(cols.is_lbu),
            cols.address_operation.addr_operation.value.map(canonical),
            cols.offset_bit.map(canonical),
            cols.memory_access.prev_value.map(canonical),
            canonical(cols.selected_limb),
            canonical(cols.selected_limb_low_byte),
            canonical(cols.selected_byte),
            canonical(cols.msb),
        );
    }
}
