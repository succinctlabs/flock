// Copyright 2025 The Binius Developers
// Copyright 2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The `PHI_8_TABLE` values are a verbatim copy from binius64
// (https://github.com/binius-zk/binius64, `crates/field/src/ghash.rs`).

//! φ_8: GF(2^8) → GF(2^128)-GHASH subfield embedding.
//!
//! `PHI_8_TABLE[v]` is the image of the F_8 element `v` (AES poly 0x11B) under
//! the field homomorphism into F_{2^128} (GHASH poly 0x87). The table is a
//! verbatim copy from binius64's `crates/field/src/ghash.rs`,
//! cross-checked with binius's proptest `test_conversion_from_aes_consistency`.
//! Verified here against the homomorphism property `phi8(a*b) = phi8(a)*phi8(b)`.

use crate::{F8, F128};

pub static PHI_8_TABLE: [F128; 256] = [
    F128 {
        lo: 0x0000000000000000,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x0000000000000001,
        hi: 0x0000000000000000,
    },
    F128 {
        lo: 0x6b8330483c2e9849,
        hi: 0x0dcb364640a222fe,
    },
    F128 {
        lo: 0x6b8330483c2e9848,
        hi: 0x0dcb364640a222fe,
    },
    F128 {
        lo: 0x7573da4a5f7710ed,
        hi: 0x3d5bd35c94646a24,
    },
    F128 {
        lo: 0x7573da4a5f7710ec,
        hi: 0x3d5bd35c94646a24,
    },
    F128 {
        lo: 0x1ef0ea02635988a4,
        hi: 0x3090e51ad4c648da,
    },
    F128 {
        lo: 0x1ef0ea02635988a5,
        hi: 0x3090e51ad4c648da,
    },
    F128 {
        lo: 0x41a12db1f974f3ac,
        hi: 0x6d58c4e181f9199f,
    },
    F128 {
        lo: 0x41a12db1f974f3ad,
        hi: 0x6d58c4e181f9199f,
    },
    F128 {
        lo: 0x2a221df9c55a6be5,
        hi: 0x6093f2a7c15b3b61,
    },
    F128 {
        lo: 0x2a221df9c55a6be4,
        hi: 0x6093f2a7c15b3b61,
    },
    F128 {
        lo: 0x34d2f7fba603e341,
        hi: 0x500317bd159d73bb,
    },
    F128 {
        lo: 0x34d2f7fba603e340,
        hi: 0x500317bd159d73bb,
    },
    F128 {
        lo: 0x5f51c7b39a2d7b08,
        hi: 0x5dc821fb553f5145,
    },
    F128 {
        lo: 0x5f51c7b39a2d7b09,
        hi: 0x5dc821fb553f5145,
    },
    F128 {
        lo: 0x5e2f716f4ede412f,
        hi: 0xa72ec17764d7ced5,
    },
    F128 {
        lo: 0x5e2f716f4ede412e,
        hi: 0xa72ec17764d7ced5,
    },
    F128 {
        lo: 0x35ac412772f0d966,
        hi: 0xaae5f7312475ec2b,
    },
    F128 {
        lo: 0x35ac412772f0d967,
        hi: 0xaae5f7312475ec2b,
    },
    F128 {
        lo: 0x2b5cab2511a951c2,
        hi: 0x9a75122bf0b3a4f1,
    },
    F128 {
        lo: 0x2b5cab2511a951c3,
        hi: 0x9a75122bf0b3a4f1,
    },
    F128 {
        lo: 0x40df9b6d2d87c98b,
        hi: 0x97be246db011860f,
    },
    F128 {
        lo: 0x40df9b6d2d87c98a,
        hi: 0x97be246db011860f,
    },
    F128 {
        lo: 0x1f8e5cdeb7aab283,
        hi: 0xca760596e52ed74a,
    },
    F128 {
        lo: 0x1f8e5cdeb7aab282,
        hi: 0xca760596e52ed74a,
    },
    F128 {
        lo: 0x740d6c968b842aca,
        hi: 0xc7bd33d0a58cf5b4,
    },
    F128 {
        lo: 0x740d6c968b842acb,
        hi: 0xc7bd33d0a58cf5b4,
    },
    F128 {
        lo: 0x6afd8694e8dda26e,
        hi: 0xf72dd6ca714abd6e,
    },
    F128 {
        lo: 0x6afd8694e8dda26f,
        hi: 0xf72dd6ca714abd6e,
    },
    F128 {
        lo: 0x017eb6dcd4f33a27,
        hi: 0xfae6e08c31e89f90,
    },
    F128 {
        lo: 0x017eb6dcd4f33a26,
        hi: 0xfae6e08c31e89f90,
    },
    F128 {
        lo: 0x5cb10fbabcf00118,
        hi: 0x4d52354a3a3d8c86,
    },
    F128 {
        lo: 0x5cb10fbabcf00119,
        hi: 0x4d52354a3a3d8c86,
    },
    F128 {
        lo: 0x37323ff280de9951,
        hi: 0x4099030c7a9fae78,
    },
    F128 {
        lo: 0x37323ff280de9950,
        hi: 0x4099030c7a9fae78,
    },
    F128 {
        lo: 0x29c2d5f0e38711f5,
        hi: 0x7009e616ae59e6a2,
    },
    F128 {
        lo: 0x29c2d5f0e38711f4,
        hi: 0x7009e616ae59e6a2,
    },
    F128 {
        lo: 0x4241e5b8dfa989bc,
        hi: 0x7dc2d050eefbc45c,
    },
    F128 {
        lo: 0x4241e5b8dfa989bd,
        hi: 0x7dc2d050eefbc45c,
    },
    F128 {
        lo: 0x1d10220b4584f2b4,
        hi: 0x200af1abbbc49519,
    },
    F128 {
        lo: 0x1d10220b4584f2b5,
        hi: 0x200af1abbbc49519,
    },
    F128 {
        lo: 0x7693124379aa6afd,
        hi: 0x2dc1c7edfb66b7e7,
    },
    F128 {
        lo: 0x7693124379aa6afc,
        hi: 0x2dc1c7edfb66b7e7,
    },
    F128 {
        lo: 0x6863f8411af3e259,
        hi: 0x1d5122f72fa0ff3d,
    },
    F128 {
        lo: 0x6863f8411af3e258,
        hi: 0x1d5122f72fa0ff3d,
    },
    F128 {
        lo: 0x03e0c80926dd7a10,
        hi: 0x109a14b16f02ddc3,
    },
    F128 {
        lo: 0x03e0c80926dd7a11,
        hi: 0x109a14b16f02ddc3,
    },
    F128 {
        lo: 0x029e7ed5f22e4037,
        hi: 0xea7cf43d5eea4253,
    },
    F128 {
        lo: 0x029e7ed5f22e4036,
        hi: 0xea7cf43d5eea4253,
    },
    F128 {
        lo: 0x691d4e9dce00d87e,
        hi: 0xe7b7c27b1e4860ad,
    },
    F128 {
        lo: 0x691d4e9dce00d87f,
        hi: 0xe7b7c27b1e4860ad,
    },
    F128 {
        lo: 0x77eda49fad5950da,
        hi: 0xd7272761ca8e2877,
    },
    F128 {
        lo: 0x77eda49fad5950db,
        hi: 0xd7272761ca8e2877,
    },
    F128 {
        lo: 0x1c6e94d79177c893,
        hi: 0xdaec11278a2c0a89,
    },
    F128 {
        lo: 0x1c6e94d79177c892,
        hi: 0xdaec11278a2c0a89,
    },
    F128 {
        lo: 0x433f53640b5ab39b,
        hi: 0x872430dcdf135bcc,
    },
    F128 {
        lo: 0x433f53640b5ab39a,
        hi: 0x872430dcdf135bcc,
    },
    F128 {
        lo: 0x28bc632c37742bd2,
        hi: 0x8aef069a9fb17932,
    },
    F128 {
        lo: 0x28bc632c37742bd3,
        hi: 0x8aef069a9fb17932,
    },
    F128 {
        lo: 0x364c892e542da376,
        hi: 0xba7fe3804b7731e8,
    },
    F128 {
        lo: 0x364c892e542da377,
        hi: 0xba7fe3804b7731e8,
    },
    F128 {
        lo: 0x5dcfb96668033b3f,
        hi: 0xb7b4d5c60bd51316,
    },
    F128 {
        lo: 0x5dcfb96668033b3e,
        hi: 0xb7b4d5c60bd51316,
    },
    F128 {
        lo: 0x95ed1f57f3632d4d,
        hi: 0x553e92e8bc0ae9a7,
    },
    F128 {
        lo: 0x95ed1f57f3632d4c,
        hi: 0x553e92e8bc0ae9a7,
    },
    F128 {
        lo: 0xfe6e2f1fcf4db504,
        hi: 0x58f5a4aefca8cb59,
    },
    F128 {
        lo: 0xfe6e2f1fcf4db505,
        hi: 0x58f5a4aefca8cb59,
    },
    F128 {
        lo: 0xe09ec51dac143da0,
        hi: 0x686541b4286e8383,
    },
    F128 {
        lo: 0xe09ec51dac143da1,
        hi: 0x686541b4286e8383,
    },
    F128 {
        lo: 0x8b1df555903aa5e9,
        hi: 0x65ae77f268cca17d,
    },
    F128 {
        lo: 0x8b1df555903aa5e8,
        hi: 0x65ae77f268cca17d,
    },
    F128 {
        lo: 0xd44c32e60a17dee1,
        hi: 0x386656093df3f038,
    },
    F128 {
        lo: 0xd44c32e60a17dee0,
        hi: 0x386656093df3f038,
    },
    F128 {
        lo: 0xbfcf02ae363946a8,
        hi: 0x35ad604f7d51d2c6,
    },
    F128 {
        lo: 0xbfcf02ae363946a9,
        hi: 0x35ad604f7d51d2c6,
    },
    F128 {
        lo: 0xa13fe8ac5560ce0c,
        hi: 0x053d8555a9979a1c,
    },
    F128 {
        lo: 0xa13fe8ac5560ce0d,
        hi: 0x053d8555a9979a1c,
    },
    F128 {
        lo: 0xcabcd8e4694e5645,
        hi: 0x08f6b313e935b8e2,
    },
    F128 {
        lo: 0xcabcd8e4694e5644,
        hi: 0x08f6b313e935b8e2,
    },
    F128 {
        lo: 0xcbc26e38bdbd6c62,
        hi: 0xf210539fd8dd2772,
    },
    F128 {
        lo: 0xcbc26e38bdbd6c63,
        hi: 0xf210539fd8dd2772,
    },
    F128 {
        lo: 0xa0415e708193f42b,
        hi: 0xffdb65d9987f058c,
    },
    F128 {
        lo: 0xa0415e708193f42a,
        hi: 0xffdb65d9987f058c,
    },
    F128 {
        lo: 0xbeb1b472e2ca7c8f,
        hi: 0xcf4b80c34cb94d56,
    },
    F128 {
        lo: 0xbeb1b472e2ca7c8e,
        hi: 0xcf4b80c34cb94d56,
    },
    F128 {
        lo: 0xd532843adee4e4c6,
        hi: 0xc280b6850c1b6fa8,
    },
    F128 {
        lo: 0xd532843adee4e4c7,
        hi: 0xc280b6850c1b6fa8,
    },
    F128 {
        lo: 0x8a63438944c99fce,
        hi: 0x9f48977e59243eed,
    },
    F128 {
        lo: 0x8a63438944c99fcf,
        hi: 0x9f48977e59243eed,
    },
    F128 {
        lo: 0xe1e073c178e70787,
        hi: 0x9283a13819861c13,
    },
    F128 {
        lo: 0xe1e073c178e70786,
        hi: 0x9283a13819861c13,
    },
    F128 {
        lo: 0xff1099c31bbe8f23,
        hi: 0xa2134422cd4054c9,
    },
    F128 {
        lo: 0xff1099c31bbe8f22,
        hi: 0xa2134422cd4054c9,
    },
    F128 {
        lo: 0x9493a98b2790176a,
        hi: 0xafd872648de27637,
    },
    F128 {
        lo: 0x9493a98b2790176b,
        hi: 0xafd872648de27637,
    },
    F128 {
        lo: 0xc95c10ed4f932c55,
        hi: 0x186ca7a286376521,
    },
    F128 {
        lo: 0xc95c10ed4f932c54,
        hi: 0x186ca7a286376521,
    },
    F128 {
        lo: 0xa2df20a573bdb41c,
        hi: 0x15a791e4c69547df,
    },
    F128 {
        lo: 0xa2df20a573bdb41d,
        hi: 0x15a791e4c69547df,
    },
    F128 {
        lo: 0xbc2fcaa710e43cb8,
        hi: 0x253774fe12530f05,
    },
    F128 {
        lo: 0xbc2fcaa710e43cb9,
        hi: 0x253774fe12530f05,
    },
    F128 {
        lo: 0xd7acfaef2ccaa4f1,
        hi: 0x28fc42b852f12dfb,
    },
    F128 {
        lo: 0xd7acfaef2ccaa4f0,
        hi: 0x28fc42b852f12dfb,
    },
    F128 {
        lo: 0x88fd3d5cb6e7dff9,
        hi: 0x7534634307ce7cbe,
    },
    F128 {
        lo: 0x88fd3d5cb6e7dff8,
        hi: 0x7534634307ce7cbe,
    },
    F128 {
        lo: 0xe37e0d148ac947b0,
        hi: 0x78ff5505476c5e40,
    },
    F128 {
        lo: 0xe37e0d148ac947b1,
        hi: 0x78ff5505476c5e40,
    },
    F128 {
        lo: 0xfd8ee716e990cf14,
        hi: 0x486fb01f93aa169a,
    },
    F128 {
        lo: 0xfd8ee716e990cf15,
        hi: 0x486fb01f93aa169a,
    },
    F128 {
        lo: 0x960dd75ed5be575d,
        hi: 0x45a48659d3083464,
    },
    F128 {
        lo: 0x960dd75ed5be575c,
        hi: 0x45a48659d3083464,
    },
    F128 {
        lo: 0x97736182014d6d7a,
        hi: 0xbf4266d5e2e0abf4,
    },
    F128 {
        lo: 0x97736182014d6d7b,
        hi: 0xbf4266d5e2e0abf4,
    },
    F128 {
        lo: 0xfcf051ca3d63f533,
        hi: 0xb2895093a242890a,
    },
    F128 {
        lo: 0xfcf051ca3d63f532,
        hi: 0xb2895093a242890a,
    },
    F128 {
        lo: 0xe200bbc85e3a7d97,
        hi: 0x8219b5897684c1d0,
    },
    F128 {
        lo: 0xe200bbc85e3a7d96,
        hi: 0x8219b5897684c1d0,
    },
    F128 {
        lo: 0x89838b806214e5de,
        hi: 0x8fd283cf3626e32e,
    },
    F128 {
        lo: 0x89838b806214e5df,
        hi: 0x8fd283cf3626e32e,
    },
    F128 {
        lo: 0xd6d24c33f8399ed6,
        hi: 0xd21aa2346319b26b,
    },
    F128 {
        lo: 0xd6d24c33f8399ed7,
        hi: 0xd21aa2346319b26b,
    },
    F128 {
        lo: 0xbd517c7bc417069f,
        hi: 0xdfd1947223bb9095,
    },
    F128 {
        lo: 0xbd517c7bc417069e,
        hi: 0xdfd1947223bb9095,
    },
    F128 {
        lo: 0xa3a19679a74e8e3b,
        hi: 0xef417168f77dd84f,
    },
    F128 {
        lo: 0xa3a19679a74e8e3a,
        hi: 0xef417168f77dd84f,
    },
    F128 {
        lo: 0xc822a6319b601672,
        hi: 0xe28a472eb7dffab1,
    },
    F128 {
        lo: 0xc822a6319b601673,
        hi: 0xe28a472eb7dffab1,
    },
    F128 {
        lo: 0x512625b1f09fa87e,
        hi: 0x93252331bf042b11,
    },
    F128 {
        lo: 0x512625b1f09fa87f,
        hi: 0x93252331bf042b11,
    },
    F128 {
        lo: 0x3aa515f9ccb13037,
        hi: 0x9eee1577ffa609ef,
    },
    F128 {
        lo: 0x3aa515f9ccb13036,
        hi: 0x9eee1577ffa609ef,
    },
    F128 {
        lo: 0x2455fffbafe8b893,
        hi: 0xae7ef06d2b604135,
    },
    F128 {
        lo: 0x2455fffbafe8b892,
        hi: 0xae7ef06d2b604135,
    },
    F128 {
        lo: 0x4fd6cfb393c620da,
        hi: 0xa3b5c62b6bc263cb,
    },
    F128 {
        lo: 0x4fd6cfb393c620db,
        hi: 0xa3b5c62b6bc263cb,
    },
    F128 {
        lo: 0x1087080009eb5bd2,
        hi: 0xfe7de7d03efd328e,
    },
    F128 {
        lo: 0x1087080009eb5bd3,
        hi: 0xfe7de7d03efd328e,
    },
    F128 {
        lo: 0x7b04384835c5c39b,
        hi: 0xf3b6d1967e5f1070,
    },
    F128 {
        lo: 0x7b04384835c5c39a,
        hi: 0xf3b6d1967e5f1070,
    },
    F128 {
        lo: 0x65f4d24a569c4b3f,
        hi: 0xc326348caa9958aa,
    },
    F128 {
        lo: 0x65f4d24a569c4b3e,
        hi: 0xc326348caa9958aa,
    },
    F128 {
        lo: 0x0e77e2026ab2d376,
        hi: 0xceed02caea3b7a54,
    },
    F128 {
        lo: 0x0e77e2026ab2d377,
        hi: 0xceed02caea3b7a54,
    },
    F128 {
        lo: 0x0f0954debe41e951,
        hi: 0x340be246dbd3e5c4,
    },
    F128 {
        lo: 0x0f0954debe41e950,
        hi: 0x340be246dbd3e5c4,
    },
    F128 {
        lo: 0x648a6496826f7118,
        hi: 0x39c0d4009b71c73a,
    },
    F128 {
        lo: 0x648a6496826f7119,
        hi: 0x39c0d4009b71c73a,
    },
    F128 {
        lo: 0x7a7a8e94e136f9bc,
        hi: 0x0950311a4fb78fe0,
    },
    F128 {
        lo: 0x7a7a8e94e136f9bd,
        hi: 0x0950311a4fb78fe0,
    },
    F128 {
        lo: 0x11f9bedcdd1861f5,
        hi: 0x049b075c0f15ad1e,
    },
    F128 {
        lo: 0x11f9bedcdd1861f4,
        hi: 0x049b075c0f15ad1e,
    },
    F128 {
        lo: 0x4ea8796f47351afd,
        hi: 0x595326a75a2afc5b,
    },
    F128 {
        lo: 0x4ea8796f47351afc,
        hi: 0x595326a75a2afc5b,
    },
    F128 {
        lo: 0x252b49277b1b82b4,
        hi: 0x549810e11a88dea5,
    },
    F128 {
        lo: 0x252b49277b1b82b5,
        hi: 0x549810e11a88dea5,
    },
    F128 {
        lo: 0x3bdba32518420a10,
        hi: 0x6408f5fbce4e967f,
    },
    F128 {
        lo: 0x3bdba32518420a11,
        hi: 0x6408f5fbce4e967f,
    },
    F128 {
        lo: 0x5058936d246c9259,
        hi: 0x69c3c3bd8eecb481,
    },
    F128 {
        lo: 0x5058936d246c9258,
        hi: 0x69c3c3bd8eecb481,
    },
    F128 {
        lo: 0x0d972a0b4c6fa966,
        hi: 0xde77167b8539a797,
    },
    F128 {
        lo: 0x0d972a0b4c6fa967,
        hi: 0xde77167b8539a797,
    },
    F128 {
        lo: 0x66141a437041312f,
        hi: 0xd3bc203dc59b8569,
    },
    F128 {
        lo: 0x66141a437041312e,
        hi: 0xd3bc203dc59b8569,
    },
    F128 {
        lo: 0x78e4f0411318b98b,
        hi: 0xe32cc527115dcdb3,
    },
    F128 {
        lo: 0x78e4f0411318b98a,
        hi: 0xe32cc527115dcdb3,
    },
    F128 {
        lo: 0x1367c0092f3621c2,
        hi: 0xeee7f36151ffef4d,
    },
    F128 {
        lo: 0x1367c0092f3621c3,
        hi: 0xeee7f36151ffef4d,
    },
    F128 {
        lo: 0x4c3607bab51b5aca,
        hi: 0xb32fd29a04c0be08,
    },
    F128 {
        lo: 0x4c3607bab51b5acb,
        hi: 0xb32fd29a04c0be08,
    },
    F128 {
        lo: 0x27b537f28935c283,
        hi: 0xbee4e4dc44629cf6,
    },
    F128 {
        lo: 0x27b537f28935c282,
        hi: 0xbee4e4dc44629cf6,
    },
    F128 {
        lo: 0x3945ddf0ea6c4a27,
        hi: 0x8e7401c690a4d42c,
    },
    F128 {
        lo: 0x3945ddf0ea6c4a26,
        hi: 0x8e7401c690a4d42c,
    },
    F128 {
        lo: 0x52c6edb8d642d26e,
        hi: 0x83bf3780d006f6d2,
    },
    F128 {
        lo: 0x52c6edb8d642d26f,
        hi: 0x83bf3780d006f6d2,
    },
    F128 {
        lo: 0x53b85b6402b1e849,
        hi: 0x7959d70ce1ee6942,
    },
    F128 {
        lo: 0x53b85b6402b1e848,
        hi: 0x7959d70ce1ee6942,
    },
    F128 {
        lo: 0x383b6b2c3e9f7000,
        hi: 0x7492e14aa14c4bbc,
    },
    F128 {
        lo: 0x383b6b2c3e9f7001,
        hi: 0x7492e14aa14c4bbc,
    },
    F128 {
        lo: 0x26cb812e5dc6f8a4,
        hi: 0x44020450758a0366,
    },
    F128 {
        lo: 0x26cb812e5dc6f8a5,
        hi: 0x44020450758a0366,
    },
    F128 {
        lo: 0x4d48b16661e860ed,
        hi: 0x49c9321635282198,
    },
    F128 {
        lo: 0x4d48b16661e860ec,
        hi: 0x49c9321635282198,
    },
    F128 {
        lo: 0x121976d5fbc51be5,
        hi: 0x140113ed601770dd,
    },
    F128 {
        lo: 0x121976d5fbc51be4,
        hi: 0x140113ed601770dd,
    },
    F128 {
        lo: 0x799a469dc7eb83ac,
        hi: 0x19ca25ab20b55223,
    },
    F128 {
        lo: 0x799a469dc7eb83ad,
        hi: 0x19ca25ab20b55223,
    },
    F128 {
        lo: 0x676aac9fa4b20b08,
        hi: 0x295ac0b1f4731af9,
    },
    F128 {
        lo: 0x676aac9fa4b20b09,
        hi: 0x295ac0b1f4731af9,
    },
    F128 {
        lo: 0x0ce99cd7989c9341,
        hi: 0x2491f6f7b4d13807,
    },
    F128 {
        lo: 0x0ce99cd7989c9340,
        hi: 0x2491f6f7b4d13807,
    },
    F128 {
        lo: 0xc4cb3ae603fc8533,
        hi: 0xc61bb1d9030ec2b6,
    },
    F128 {
        lo: 0xc4cb3ae603fc8532,
        hi: 0xc61bb1d9030ec2b6,
    },
    F128 {
        lo: 0xaf480aae3fd21d7a,
        hi: 0xcbd0879f43ace048,
    },
    F128 {
        lo: 0xaf480aae3fd21d7b,
        hi: 0xcbd0879f43ace048,
    },
    F128 {
        lo: 0xb1b8e0ac5c8b95de,
        hi: 0xfb406285976aa892,
    },
    F128 {
        lo: 0xb1b8e0ac5c8b95df,
        hi: 0xfb406285976aa892,
    },
    F128 {
        lo: 0xda3bd0e460a50d97,
        hi: 0xf68b54c3d7c88a6c,
    },
    F128 {
        lo: 0xda3bd0e460a50d96,
        hi: 0xf68b54c3d7c88a6c,
    },
    F128 {
        lo: 0x856a1757fa88769f,
        hi: 0xab43753882f7db29,
    },
    F128 {
        lo: 0x856a1757fa88769e,
        hi: 0xab43753882f7db29,
    },
    F128 {
        lo: 0xeee9271fc6a6eed6,
        hi: 0xa688437ec255f9d7,
    },
    F128 {
        lo: 0xeee9271fc6a6eed7,
        hi: 0xa688437ec255f9d7,
    },
    F128 {
        lo: 0xf019cd1da5ff6672,
        hi: 0x9618a6641693b10d,
    },
    F128 {
        lo: 0xf019cd1da5ff6673,
        hi: 0x9618a6641693b10d,
    },
    F128 {
        lo: 0x9b9afd5599d1fe3b,
        hi: 0x9bd39022563193f3,
    },
    F128 {
        lo: 0x9b9afd5599d1fe3a,
        hi: 0x9bd39022563193f3,
    },
    F128 {
        lo: 0x9ae44b894d22c41c,
        hi: 0x613570ae67d90c63,
    },
    F128 {
        lo: 0x9ae44b894d22c41d,
        hi: 0x613570ae67d90c63,
    },
    F128 {
        lo: 0xf1677bc1710c5c55,
        hi: 0x6cfe46e8277b2e9d,
    },
    F128 {
        lo: 0xf1677bc1710c5c54,
        hi: 0x6cfe46e8277b2e9d,
    },
    F128 {
        lo: 0xef9791c31255d4f1,
        hi: 0x5c6ea3f2f3bd6647,
    },
    F128 {
        lo: 0xef9791c31255d4f0,
        hi: 0x5c6ea3f2f3bd6647,
    },
    F128 {
        lo: 0x8414a18b2e7b4cb8,
        hi: 0x51a595b4b31f44b9,
    },
    F128 {
        lo: 0x8414a18b2e7b4cb9,
        hi: 0x51a595b4b31f44b9,
    },
    F128 {
        lo: 0xdb456638b45637b0,
        hi: 0x0c6db44fe62015fc,
    },
    F128 {
        lo: 0xdb456638b45637b1,
        hi: 0x0c6db44fe62015fc,
    },
    F128 {
        lo: 0xb0c656708878aff9,
        hi: 0x01a68209a6823702,
    },
    F128 {
        lo: 0xb0c656708878aff8,
        hi: 0x01a68209a6823702,
    },
    F128 {
        lo: 0xae36bc72eb21275d,
        hi: 0x3136671372447fd8,
    },
    F128 {
        lo: 0xae36bc72eb21275c,
        hi: 0x3136671372447fd8,
    },
    F128 {
        lo: 0xc5b58c3ad70fbf14,
        hi: 0x3cfd515532e65d26,
    },
    F128 {
        lo: 0xc5b58c3ad70fbf15,
        hi: 0x3cfd515532e65d26,
    },
    F128 {
        lo: 0x987a355cbf0c842b,
        hi: 0x8b49849339334e30,
    },
    F128 {
        lo: 0x987a355cbf0c842a,
        hi: 0x8b49849339334e30,
    },
    F128 {
        lo: 0xf3f9051483221c62,
        hi: 0x8682b2d579916cce,
    },
    F128 {
        lo: 0xf3f9051483221c63,
        hi: 0x8682b2d579916cce,
    },
    F128 {
        lo: 0xed09ef16e07b94c6,
        hi: 0xb61257cfad572414,
    },
    F128 {
        lo: 0xed09ef16e07b94c7,
        hi: 0xb61257cfad572414,
    },
    F128 {
        lo: 0x868adf5edc550c8f,
        hi: 0xbbd96189edf506ea,
    },
    F128 {
        lo: 0x868adf5edc550c8e,
        hi: 0xbbd96189edf506ea,
    },
    F128 {
        lo: 0xd9db18ed46787787,
        hi: 0xe6114072b8ca57af,
    },
    F128 {
        lo: 0xd9db18ed46787786,
        hi: 0xe6114072b8ca57af,
    },
    F128 {
        lo: 0xb25828a57a56efce,
        hi: 0xebda7634f8687551,
    },
    F128 {
        lo: 0xb25828a57a56efcf,
        hi: 0xebda7634f8687551,
    },
    F128 {
        lo: 0xaca8c2a7190f676a,
        hi: 0xdb4a932e2cae3d8b,
    },
    F128 {
        lo: 0xaca8c2a7190f676b,
        hi: 0xdb4a932e2cae3d8b,
    },
    F128 {
        lo: 0xc72bf2ef2521ff23,
        hi: 0xd681a5686c0c1f75,
    },
    F128 {
        lo: 0xc72bf2ef2521ff22,
        hi: 0xd681a5686c0c1f75,
    },
    F128 {
        lo: 0xc6554433f1d2c504,
        hi: 0x2c6745e45de480e5,
    },
    F128 {
        lo: 0xc6554433f1d2c505,
        hi: 0x2c6745e45de480e5,
    },
    F128 {
        lo: 0xadd6747bcdfc5d4d,
        hi: 0x21ac73a21d46a21b,
    },
    F128 {
        lo: 0xadd6747bcdfc5d4c,
        hi: 0x21ac73a21d46a21b,
    },
    F128 {
        lo: 0xb3269e79aea5d5e9,
        hi: 0x113c96b8c980eac1,
    },
    F128 {
        lo: 0xb3269e79aea5d5e8,
        hi: 0x113c96b8c980eac1,
    },
    F128 {
        lo: 0xd8a5ae31928b4da0,
        hi: 0x1cf7a0fe8922c83f,
    },
    F128 {
        lo: 0xd8a5ae31928b4da1,
        hi: 0x1cf7a0fe8922c83f,
    },
    F128 {
        lo: 0x87f4698208a636a8,
        hi: 0x413f8105dc1d997a,
    },
    F128 {
        lo: 0x87f4698208a636a9,
        hi: 0x413f8105dc1d997a,
    },
    F128 {
        lo: 0xec7759ca3488aee1,
        hi: 0x4cf4b7439cbfbb84,
    },
    F128 {
        lo: 0xec7759ca3488aee0,
        hi: 0x4cf4b7439cbfbb84,
    },
    F128 {
        lo: 0xf287b3c857d12645,
        hi: 0x7c6452594879f35e,
    },
    F128 {
        lo: 0xf287b3c857d12644,
        hi: 0x7c6452594879f35e,
    },
    F128 {
        lo: 0x990483806bffbe0c,
        hi: 0x71af641f08dbd1a0,
    },
    F128 {
        lo: 0x990483806bffbe0d,
        hi: 0x71af641f08dbd1a0,
    },
];

#[inline]
pub fn phi8(a: F8) -> F128 {
    PHI_8_TABLE[a.0 as usize]
}

#[cfg(test)]
mod tests {
    use crate::phi8::{F8, F128, phi8};

    #[test]
    fn zero_and_one_map_correctly() {
        assert_eq!(phi8(F8::ZERO), F128::ZERO);
        assert_eq!(phi8(F8::ONE), F128::ONE);
    }

    #[test]
    fn homomorphism_full() {
        // Exhaustive check: φ(a·b) = φ(a)·φ(b) and φ(a+b) = φ(a)+φ(b)
        // for all 65536 ordered pairs in F_8.
        for a in 0u8..=255 {
            for b in 0u8..=255 {
                let fa = F8(a);
                let fb = F8(b);
                let lhs_mul = phi8(fa * fb);
                let rhs_mul = phi8(fa) * phi8(fb);
                assert_eq!(lhs_mul, rhs_mul, "mul mismatch at a={a}, b={b}");

                let lhs_add = phi8(fa + fb);
                let rhs_add = phi8(fa) + phi8(fb);
                assert_eq!(lhs_add, rhs_add, "add mismatch at a={a}, b={b}");
            }
        }
    }
}
