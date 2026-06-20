use crate::{
    nnue::network::{L1_SIZE, PSQT_FEATURES, feature::PsqtFeatureIndex},
    util::Align,
};

/// Pre-activations of l0’s output.
pub struct Accumulator {
    pub halves: [Align<[i16; L1_SIZE]>; 2],
}

#[allow(clippy::inline_always)]
#[inline(always)]
unsafe fn slice_to_aligned<T>(slice: &[T]) -> &Align<[T; L1_SIZE]> {
    debug_assert_eq!(slice.len(), L1_SIZE);
    // don't immediately cast to Align64, as we want to check the alignment first.
    let ptr = slice.as_ptr();
    debug_assert_eq!(ptr.align_offset(64), 0);
    // Safety: alignments are sensible, so we can safely cast.
    #[allow(clippy::cast_ptr_alignment)]
    unsafe {
        &*ptr.cast()
    }
}

mod simd {
    use arrayvec::ArrayVec;

    use super::{Align, L1_SIZE, PSQT_FEATURES, PsqtFeatureIndex, slice_to_aligned};
    use crate::{
        chess::{
            board::{Board, movegen::attacks_by_type},
            piece::{Colour, PieceType},
            types::Square,
        },
        nnue::{
            network::{
                AUX_FEATURES, AuxUpdateBuffer, PAWN_TUPLE_FEATURES,
                feature::{pawn_pawn_index, threat_index},
                pawn_updates::PAWN_PAWN_MASKS,
            },
            simd::{self, I16_CHUNK},
        },
    };

    /// Apply add/subtract PSQT updates in place.
    pub fn vector_update_inplace_psqt(
        input: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        adds: &[PsqtFeatureIndex],
        subs: &[PsqtFeatureIndex],
    ) {
        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;
        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            let mut add_blocks = ArrayVec::<_, 32>::new();
            let mut sub_blocks = ArrayVec::<_, 32>::new();
            for &add_index in adds {
                let add_index = add_index.index() * L1_SIZE;
                add_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(add_index..add_index + L1_SIZE),
                ));
            }
            for &sub_index in subs {
                let sub_index = sub_index.index() * L1_SIZE;
                sub_blocks.push(slice_to_aligned(
                    bucket.get_unchecked(sub_index..sub_index + L1_SIZE),
                ));
            }
            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let src = input.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    *reg = simd::load_i16(src);
                }
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = sub_block.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::sub_i16(*reg, simd::load_i16(src));
                    }
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = add_block.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_i16(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = input.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    /// Apply add/subtract updates in place.
    pub fn vector_update_aux(
        src_acc: &Align<[i16; L1_SIZE]>,
        dst_acc: &mut Align<[i16; L1_SIZE]>,
        weights: &Align<[i8; AUX_FEATURES * L1_SIZE]>,
        updates: &AuxUpdateBuffer,
        king: Square,
        colour: Colour,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;

        if updates.add.is_empty() && updates.sub.is_empty() && updates.afore == updates.after {
            dst_acc.copy_from_slice(&**src_acc);
            return;
        }

        // SAFETY: we never hold multiple mutable references, we never mutate immutable memory,
        // we use iterators to ensure that we're staying in-bounds, etc.
        unsafe {
            let mut add_blocks = ArrayVec::<u32, 192>::new();
            let mut sub_blocks = ArrayVec::<u32, 192>::new();
            add_threat_indexes(updates, king, colour, &mut add_blocks, &mut sub_blocks);
            add_pawn_pawn_indexes(updates, king, colour, &mut add_blocks, &mut sub_blocks);
            for &offset in &add_blocks {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
            for &offset in &sub_blocks {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let src = src_acc.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    *reg = simd::load_i16(src);
                }
                // todo: is load_extend_i8 the fastest way to do this?
                // check if the compiler is smart enough to load in a sensible way.
                for &sub_block in &sub_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(sub_block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::sub_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for &add_block in &add_blocks {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(add_block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = dst_acc.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    fn add_threat_indexes(
        updates: &AuxUpdateBuffer,
        king: Square,
        colour: Colour,
        adds: &mut ArrayVec<u32, 192>,
        subs: &mut ArrayVec<u32, 192>,
    ) {
        #![allow(clippy::cast_possible_truncation)]
        // Safety: Inputs are equal in size to outputs.
        unsafe {
            for &idx in &updates.add {
                let (good, idx) = idx.index(colour, king);
                let len = adds.len();
                debug_assert!(len < adds.capacity(), "OOB write");
                let loc = adds.as_mut_ptr().add(len);
                // offset to make space for pawn features
                *loc = (PAWN_TUPLE_FEATURES as u32)
                    .wrapping_add(idx)
                    .wrapping_mul(L1_SIZE as u32);
                adds.set_len(len + usize::from(good));
            }
            for &idx in &updates.sub {
                let (good, idx) = idx.index(colour, king);
                let len = subs.len();
                debug_assert!(len < subs.capacity(), "OOB write");
                let loc = subs.as_mut_ptr().add(len);
                // offset to make space for pawn features
                *loc = (PAWN_TUPLE_FEATURES as u32)
                    .wrapping_add(idx)
                    .wrapping_mul(L1_SIZE as u32);
                subs.set_len(len + usize::from(good));
            }
        }
    }

    // #[cfg(target_feature = "avx512vbmi")]
    // fn add_pawn_pawn_indexes(
    //     buffer: &AuxUpdateBuffer,
    //     king: Square,
    //     colour: Colour,
    //     adds: &mut ArrayVec<u32, 192>,
    //     subs: &mut ArrayVec<u32, 192>,
    // ) {
    //     todo!()
    // }

    // #[cfg(not(target_feature = "avx512vbmi"))]
    fn add_pawn_pawn_indexes(
        buffer: &AuxUpdateBuffer,
        king: Square,
        colour: Colour,
        adds: &mut ArrayVec<u32, 192>,
        subs: &mut ArrayVec<u32, 192>,
    ) {
        #![allow(clippy::cast_possible_truncation)]

        let mut afore_remaining = buffer.afore[Colour::White] | buffer.afore[Colour::Black];
        let mut after_remaining = buffer.after[Colour::White] | buffer.after[Colour::Black];

        let add = [
            buffer.after[Colour::White] & !buffer.afore[Colour::White],
            buffer.after[Colour::Black] & !buffer.afore[Colour::Black],
        ];
        let sub = [
            buffer.afore[Colour::White] & !buffer.after[Colour::White],
            buffer.afore[Colour::Black] & !buffer.after[Colour::Black],
        ];

        // Safety: Inputs are equal in size to outputs.
        unsafe {
            for pawn_colour in Colour::all() {
                for a in add[pawn_colour] {
                    after_remaining &= !a.as_set();

                    let mask = PAWN_PAWN_MASKS[a.file()] & after_remaining;

                    for b in buffer.after[Colour::White] & mask {
                        let index = pawn_pawn_index(colour, king, pawn_colour, a, Colour::White, b);
                        adds.push_unchecked(u32::from(index) * L1_SIZE as u32);
                    }

                    for b in buffer.after[Colour::Black] & mask {
                        let index = pawn_pawn_index(colour, king, pawn_colour, a, Colour::Black, b);
                        adds.push_unchecked(u32::from(index) * L1_SIZE as u32);
                    }
                }

                for a in sub[pawn_colour] {
                    afore_remaining &= !a.as_set();

                    let mask = PAWN_PAWN_MASKS[a.file()] & afore_remaining;

                    for b in buffer.afore[Colour::White] & mask {
                        let index = pawn_pawn_index(colour, king, pawn_colour, a, Colour::White, b);
                        subs.push_unchecked(u32::from(index) * L1_SIZE as u32);
                    }

                    for b in buffer.afore[Colour::Black] & mask {
                        let index = pawn_pawn_index(colour, king, pawn_colour, a, Colour::Black, b);
                        subs.push_unchecked(u32::from(index) * L1_SIZE as u32);
                    }
                }
            }
        }
    }

    pub fn refresh_aux(
        weights: &Align<[i8; AUX_FEATURES * L1_SIZE]>,
        acc: &mut Align<[i16; L1_SIZE]>,
        board: &Board,
        colour: Colour,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;

        let bbs = &board.state.bbs;
        let occ = bbs.occupied();
        let king = board.state.bbs.king_sq(colour);
        let bb = occ & !bbs.pieces[PieceType::King];

        let mut indexes = ArrayVec::<u32, 256>::new();

        // Safety: We know this routine never produces more than 256 features.
        unsafe {
            #![allow(clippy::cast_possible_truncation)]

            // add threat features
            for from in bb {
                let attacker = board.state.mailbox[from].unwrap();
                let threats =
                    occ & attacks_by_type(attacker, from, occ) & !bbs.pieces[PieceType::King];
                for to in threats {
                    let victim = board.state.mailbox[to].unwrap();
                    let (good, feature) = threat_index(colour, king, attacker, victim, from, to);
                    let len = indexes.len();
                    debug_assert!(len < indexes.capacity(), "OOB write");
                    let loc = indexes.as_mut_ptr().add(len);
                    // feature offset to make space for pawns.
                    *loc = (PAWN_TUPLE_FEATURES as u32)
                        .wrapping_add(feature)
                        .wrapping_mul(L1_SIZE as u32);
                    indexes.set_len(len + usize::from(good));
                }
            }

            // add pawn-pawn features
            let our_pawns = bbs.pieces[PieceType::Pawn] & bbs.colours[colour];
            let their_pawns = bbs.pieces[PieceType::Pawn] & bbs.colours[!colour];

            let mut our_pawns_iter = our_pawns.into_iter();
            let mut their_pawns_iter = their_pawns.into_iter();

            while let Some(a) = our_pawns_iter.next() {
                let mask = PAWN_PAWN_MASKS[a.file()];
                for b in our_pawns_iter.remaining() & mask {
                    let index = u32::from(pawn_pawn_index(colour, king, colour, a, colour, b));
                    indexes.push_unchecked(index * L1_SIZE as u32);
                }
                for b in their_pawns & mask {
                    let index = u32::from(pawn_pawn_index(colour, king, colour, a, !colour, b));
                    indexes.push_unchecked(index * L1_SIZE as u32);
                }
            }

            while let Some(a) = their_pawns_iter.next() {
                let mask = PAWN_PAWN_MASKS[a.file()];
                for b in their_pawns_iter.remaining() & mask {
                    let index = u32::from(pawn_pawn_index(colour, king, !colour, a, !colour, b));
                    indexes.push_unchecked(index * L1_SIZE as u32);
                }
            }

            for &offset in &indexes {
                #[cfg(target_arch = "x86_64")]
                _mm_prefetch(
                    (*weights).as_ptr().add(offset as usize).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }

            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for reg in &mut registers {
                    *reg = simd::zero_i16();
                }
                for &block in &indexes {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = (*weights)
                            .as_ptr()
                            .add(block as usize + unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_extend_i8(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = acc.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    #[inline]
    unsafe fn psqt_block(
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        feature: PsqtFeatureIndex,
    ) -> &Align<[i16; L1_SIZE]> {
        let offset = feature.index() * L1_SIZE;
        unsafe { slice_to_aligned(bucket.get_unchecked(offset..offset + L1_SIZE)) }
    }

    #[inline]
    unsafe fn vector_add_sub_tiled<const ADDS: usize, const SUBS: usize>(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        adds: [&Align<[i16; L1_SIZE]>; ADDS],
        subs: [&Align<[i16; L1_SIZE]>; SUBS],
    ) {
        const REGISTERS: usize = 16;
        const UNROLL: usize = I16_CHUNK * REGISTERS;
        unsafe {
            let mut registers = [simd::zero_i16(); REGISTERS];
            for i in 0..L1_SIZE / UNROLL {
                let unroll_offset = i * UNROLL;
                for (r_idx, reg) in registers.iter_mut().enumerate() {
                    let src = input.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    *reg = simd::load_i16(src);
                }
                for sub in &subs {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = sub.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::sub_i16(*reg, simd::load_i16(src));
                    }
                }
                for add in &adds {
                    for (r_idx, reg) in registers.iter_mut().enumerate() {
                        let src = add.as_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                        *reg = simd::add_i16(*reg, simd::load_i16(src));
                    }
                }
                for (r_idx, reg) in registers.iter().enumerate() {
                    let dst = output.as_mut_ptr().add(unroll_offset + r_idx * I16_CHUNK);
                    simd::store_i16(dst, *reg);
                }
            }
        }
    }

    /// Move a PSQT feature from one square to another.
    pub fn vector_add_sub_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add: PsqtFeatureIndex,
        sub: PsqtFeatureIndex,
    ) {
        unsafe {
            vector_add_sub_tiled(
                input,
                output,
                [psqt_block(bucket, add)],
                [psqt_block(bucket, sub)],
            );
        }
    }

    /// Subtract two PSQT features and add one PSQT feature all at once.
    pub fn vector_add_sub2_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add: PsqtFeatureIndex,
        sub1: PsqtFeatureIndex,
        sub2: PsqtFeatureIndex,
    ) {
        unsafe {
            vector_add_sub_tiled(
                input,
                output,
                [psqt_block(bucket, add)],
                [psqt_block(bucket, sub1), psqt_block(bucket, sub2)],
            );
        }
    }

    /// Add two PSQT features and subtract two PSQT features all at once.
    pub fn vector_add2_sub2_psqt(
        input: &Align<[i16; L1_SIZE]>,
        output: &mut Align<[i16; L1_SIZE]>,
        bucket: &Align<[i16; PSQT_FEATURES * L1_SIZE]>,
        add1: PsqtFeatureIndex,
        add2: PsqtFeatureIndex,
        sub1: PsqtFeatureIndex,
        sub2: PsqtFeatureIndex,
    ) {
        unsafe {
            vector_add_sub_tiled(
                input,
                output,
                [psqt_block(bucket, add1), psqt_block(bucket, add2)],
                [psqt_block(bucket, sub1), psqt_block(bucket, sub2)],
            );
        }
    }
}

pub use simd::*;
