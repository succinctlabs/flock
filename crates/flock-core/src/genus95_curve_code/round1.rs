//! Production round-1 AG-code URM kernel — eq-folded encode·product·fold, fused
//! AB+C in a single pass over the witness.
//!
//! Lifted verbatim from `benches/urm_bitslice.rs` (auto-generated genus-95
//! by-point encode `M` (160×64) + Hasse/Leibniz product + GHASH fold), which is
//! cross-checked byte-identical to a scalar reference. `aarch64`/NEON only (the
//! production target); the packed-witness input path, the eq-from-`r` derivation,
//! the 160→222 layout bridge, the `product_code_message` cross-check, and Metal
//! land in later M1/M3 steps.
//!
//! Output is the 160 by-point FRESH product-code coordinates
//! (D¹·64 | D²·64 | D³·32, two garbage rows at the D³ points 28/31). The 64
//! systematic "value" coordinates are the raw witness product; the verifier
//! reconstructs them from the zerocheck identity, so they are not emitted here.
#![allow(dead_code)] // WIP (M1): packed input + entry points land incrementally.

use crate::field::{F128, F256Unreduced};
use std::arch::aarch64::*;
use std::sync::OnceLock;

use super::messages::BaseMessage;
use super::product::extended_base_product_message;

const M_MASK: [u64; 160] = [
    0x12524642c1f3a14e,
    0x76d50d081b8d65ee,
    0xf73d75bccf36d76f,
    0x9e68e026a9537209,
    0x5b083dc8a0e78433,
    0x845267390dbb374e,
    0xa0e0dc4e833cdf6a,
    0xa8519d5b90472d03,
    0xa0acf147dd3c1bfe,
    0xdf039deab59a9e8a,
    0x987786f8622bd66e,
    0xc9ac91b7d43a7d61,
    0x9e819bcd70937a1a,
    0xded2c75fd805f35e,
    0xcdeef1cb40a34f7f,
    0xe0698a16add8aef4,
    0x4ce510edde796eb6,
    0x0834fea9996429a8,
    0x615bfe59900b20a8,
    0xd3ea80e44779f7d9,
    0x2af4458ba968197a,
    0xa11a58ff214a6e97,
    0xb1ef586886b5eb80,
    0xdcf4bc14c098707c,
    0xedb536fd0b3cd73d,
    0x4d83336dc2f9e2fb,
    0x6b13363596959d70,
    0x133e284eee8716ad,
    0x814fff25c2273c92,
    0xbe8003e902d833ad,
    0xc1759c713c484043,
    0x57d0f51b66d2dae9,
    0xa00d02a7622f1544,
    0x09579bfe34ba8321,
    0xeb8f2adfb0ce6866,
    0x4dd54c89e9ab01f3,
    0x7da702e38c58bf99,
    0xe61f7a5a952577c6,
    0x5c2629f1071031ed,
    0x323e0eb2b1cda61d,
    0x99f2bb0459231fcf,
    0xf18a308183a1d7a0,
    0x6815a0888aa7be39,
    0xc06127c2acd685aa,
    0x2707e4c2ac8c042b,
    0x4eaebda739196db2,
    0xbb6182ab9fbf321d,
    0x868f6047d3545027,
    0x6ae400a5508a8541,
    0xc430e42519972282,
    0x667ec05cc01f4021,
    0xc948c5ff05d9491b,
    0x3ef9d2543d6235a6,
    0xc205e268019e3a95,
    0x3d931d0808f85563,
    0x27952b33fa0449b1,
    0x65b8e99899701e6e,
    0x6d0a9782b60bdc38,
    0x2d0a0bbc8bf1a32c,
    0x80edd00cfe20cb2f,
    0xb3397a8297f42362,
    0xe88d31f77d791732,
    0x09f3492c3934554c,
    0xbaa01582619d2304,
    0x03db7670c07f1021,
    0x89d745db3b46f639,
    0x5eac57a07c612ebd,
    0xadac949c40aeddbd,
    0x9be41f7e52f1d7f5,
    0xed1b201aaaf701e1,
    0xe300aecfb92c3611,
    0xdb7d13d6699bc787,
    0x2df0d02c8fa1f3a5,
    0x20d102c0fc499d63,
    0xd027f4a9634f6d6c,
    0x72fcdce5eaadf65f,
    0x21b70e2092b35595,
    0xcb4208176fb386b4,
    0x5ca91fcb8c914ed4,
    0x27b801b0fbbcc50a,
    0xd81bfbb89e5d105f,
    0xdfa5896d8126e1f9,
    0x327438bc95cd6321,
    0x953040397fb36772,
    0xdf000d3705349a14,
    0xff5a57f967a4169a,
    0x08b1b9231b16def5,
    0x742e29bebc4368b1,
    0xc0cb49fcb8e51702,
    0x7f671651ef70618a,
    0x1f0789381f86678a,
    0xdd1597dbac6efa4c,
    0x62ecf4921f6d4fdf,
    0x3d25320dd597b515,
    0x4eef040fd23e0032,
    0x7def0bfcde02333e,
    0xf5af4d40073e9fbf,
    0x1424caaed5457134,
    0x92213aafedd3022a,
    0x0ee403ffd425f8ef,
    0xb5b81d8860b58f8f,
    0x35e47dda382fc938,
    0x08a5af5377cecd9d,
    0xfdc5006a279da1fd,
    0xf25bcb8f31f204f7,
    0x11fea1429fad88b5,
    0xb5b68c5429d313e0,
    0x52ae584f54076e04,
    0xfd98086ee364e58f,
    0x3d9bc752ec97eab3,
    0xc83bfeae4a37bc83,
    0xb1c732c58e31a55b,
    0x4cd4edf2dbdd6492,
    0x5d9ca5dc3aaabad6,
    0xaa7144ff4f877d26,
    0xd3428b6444426402,
    0x424e4795fdc524a7,
    0xf77287013acccd4f,
    0xe421d89c3d05e198,
    0x871b1dc0c700e792,
    0x339427132bbcdece,
    0x4fc7448e83da674a,
    0x71d6c0629c9d3c26,
    0x693e1bb611767d0d,
    0x5875f6cc4ab99d91,
    0x6664282f36920f57,
    0x9a548e2cf05450f1,
    0x5dda30c613e0a2a8,
    0x430dfb71bf88cb91,
    0x3391a8ef1bdd82d5,
    0xc0a1f1ec2274247c,
    0x874cd0679999a0a2,
    0x81b573527709cf3d,
    0x70a85e8e3be1a207,
    0xdfa2c2215182316b,
    0xbeb61962146c5a58,
    0xc10e9d505eb47847,
    0x3ef1916f6177b848,
    0x93d94c7048ff4c7b,
    0x6efb6e6fcb271242,
    0x24cb1f1f1421103c,
    0xe708e3d02421dff0,
    0x001094026f6c6b18,
    0x64ab1c48b92d6047,
    0x24b70eb327efa9cb,
    0x422268e54d75c092,
    0x924df2bbe61fb08f,
    0x22de9e254179c63b,
    0x023938a69dcedf51,
    0x96b270e3788c28fd,
    0x0bf6a7aa5d0e4f34,
    0x4d39a4fea9c4a623,
    0x70f67229ce1f5434,
    0x8744ccf01268a59e,
    0xd141395fe76b5f62,
    0x6a53242b5fb6e88c,
    0x0000000000000000,
    0xe155882d033de134,
    0x2aedc3a42a2c1a07,
    0x0000000000000000,
];

#[inline(never)]
fn encode_slp(inp: &[uint8x16_t; 64], out: &mut [uint8x16_t; 160]) {
    unsafe {
        let s64 = veorq_u8(inp[8], inp[24]);
        let s65 = veorq_u8(inp[18], inp[35]);
        let s66 = veorq_u8(inp[53], inp[55]);
        let s67 = veorq_u8(inp[19], inp[50]);
        let s68 = veorq_u8(inp[10], inp[39]);
        let s69 = veorq_u8(inp[1], inp[27]);
        let s70 = veorq_u8(inp[20], inp[37]);
        let s71 = veorq_u8(inp[21], inp[56]);
        let s72 = veorq_u8(inp[5], inp[31]);
        let s73 = veorq_u8(inp[13], inp[16]);
        let s74 = veorq_u8(inp[38], inp[62]);
        let s75 = veorq_u8(inp[25], inp[48]);
        let s76 = veorq_u8(inp[28], inp[43]);
        let s77 = veorq_u8(inp[14], inp[44]);
        let s78 = veorq_u8(inp[42], inp[52]);
        let s79 = veorq_u8(inp[3], inp[58]);
        let s80 = veorq_u8(inp[9], inp[63]);
        let s81 = veorq_u8(inp[0], inp[23]);
        let s82 = veorq_u8(inp[12], inp[59]);
        let s83 = veorq_u8(inp[34], inp[54]);
        let s84 = veorq_u8(inp[11], inp[33]);
        let s85 = veorq_u8(inp[2], inp[60]);
        let s86 = veorq_u8(inp[4], inp[32]);
        let s87 = veorq_u8(inp[30], inp[57]);
        let s88 = veorq_u8(inp[36], inp[40]);
        let s89 = veorq_u8(inp[7], inp[15]);
        let s90 = veorq_u8(inp[17], inp[49]);
        let s91 = veorq_u8(inp[51], inp[61]);
        let s92 = veorq_u8(inp[46], inp[47]);
        let s93 = veorq_u8(inp[22], inp[29]);
        let s94 = veorq_u8(inp[26], inp[41]);
        let s95 = veorq_u8(inp[6], inp[45]);
        let s96 = veorq_u8(s66, s67);
        let s97 = veorq_u8(s65, s73);
        let s98 = veorq_u8(s69, s78);
        let s99 = veorq_u8(s64, s77);
        let s100 = veorq_u8(s71, s80);
        let s101 = veorq_u8(s68, s74);
        let s102 = veorq_u8(s70, s82);
        let s103 = veorq_u8(inp[26], s84);
        let s104 = veorq_u8(inp[57], s76);
        let s105 = veorq_u8(inp[45], s72);
        let s106 = veorq_u8(s83, s87);
        let s107 = veorq_u8(inp[62], s68);
        let s108 = veorq_u8(inp[24], inp[33]);
        let s109 = veorq_u8(inp[9], s81);
        let s110 = veorq_u8(inp[5], s79);
        let s111 = veorq_u8(inp[14], s64);
        let s112 = veorq_u8(inp[61], s75);
        let s113 = veorq_u8(s76, s88);
        let s114 = veorq_u8(inp[56], s90);
        let s115 = veorq_u8(inp[3], inp[21]);
        let s116 = veorq_u8(inp[17], inp[48]);
        let s117 = veorq_u8(inp[2], s91);
        let s118 = veorq_u8(inp[34], s86);
        let s119 = veorq_u8(inp[4], s89);
        let s120 = veorq_u8(inp[7], inp[50]);
        let s121 = veorq_u8(inp[10], s85);
        let s122 = veorq_u8(inp[16], inp[43]);
        let s123 = veorq_u8(inp[30], s75);
        let s124 = veorq_u8(inp[27], s65);
        let s125 = veorq_u8(inp[39], s85);
        let s126 = veorq_u8(inp[20], inp[23]);
        let s127 = veorq_u8(inp[32], inp[63]);
        let s128 = veorq_u8(inp[28], s69);
        let s129 = veorq_u8(inp[40], inp[47]);
        let s130 = veorq_u8(inp[35], s83);
        let s131 = veorq_u8(inp[18], inp[49]);
        let s132 = veorq_u8(inp[44], s94);
        let s133 = veorq_u8(inp[13], s74);
        let s134 = veorq_u8(inp[38], inp[46]);
        let s135 = veorq_u8(inp[41], s78);
        let s136 = veorq_u8(inp[55], s67);
        let s137 = veorq_u8(inp[34], inp[53]);
        let s138 = veorq_u8(inp[23], s70);
        let s139 = veorq_u8(inp[6], s66);
        let s140 = veorq_u8(inp[12], s92);
        let s141 = veorq_u8(inp[8], inp[16]);
        let s142 = veorq_u8(inp[36], s93);
        let s143 = veorq_u8(s84, s94);
        let s144 = veorq_u8(inp[52], s71);
        let s145 = veorq_u8(inp[1], inp[22]);
        let s146 = veorq_u8(inp[6], inp[54]);
        let s147 = veorq_u8(s72, s95);
        let s148 = veorq_u8(inp[0], inp[51]);
        let s149 = veorq_u8(inp[60], s79);
        let s150 = veorq_u8(inp[42], inp[59]);
        let s151 = veorq_u8(inp[37], s82);
        let s152 = veorq_u8(inp[31], inp[49]);
        let s153 = veorq_u8(inp[25], inp[29]);
        let s154 = veorq_u8(inp[58], s73);
        let s155 = veorq_u8(inp[15], s88);
        let s156 = veorq_u8(s91, s100);
        let s157 = veorq_u8(inp[51], s81);
        let s158 = veorq_u8(inp[2], inp[10]);
        let s159 = veorq_u8(inp[38], inp[61]);
        let s160 = veorq_u8(inp[53], s67);
        let s161 = veorq_u8(s77, s96);
        let s162 = veorq_u8(inp[35], s77);
        let s163 = veorq_u8(inp[19], inp[22]);
        let s164 = veorq_u8(inp[27], s86);
        let s165 = veorq_u8(inp[15], s115);
        let s166 = veorq_u8(inp[31], inp[45]);
        let s167 = veorq_u8(inp[1], inp[40]);
        let s168 = veorq_u8(inp[21], inp[46]);
        let s169 = veorq_u8(inp[32], s88);
        let s170 = veorq_u8(inp[29], inp[48]);
        let s171 = veorq_u8(inp[13], inp[47]);
        let s172 = veorq_u8(inp[37], inp[59]);
        let s173 = veorq_u8(inp[0], inp[52]);
        let s174 = veorq_u8(inp[43], inp[63]);
        let s175 = veorq_u8(inp[54], s87);
        let s176 = veorq_u8(inp[55], inp[60]);
        let s177 = veorq_u8(inp[52], s104);
        let s178 = veorq_u8(s97, s112);
        let s179 = veorq_u8(s98, s146);
        let s180 = veorq_u8(inp[0], inp[50]);
        let s181 = veorq_u8(s65, s75);
        let s182 = veorq_u8(s99, s107);
        let s183 = veorq_u8(inp[57], s92);
        let s184 = veorq_u8(inp[4], s93);
        let s185 = veorq_u8(inp[19], s80);
        let s186 = veorq_u8(inp[25], inp[42]);
        let s187 = veorq_u8(inp[18], s96);
        let s188 = veorq_u8(inp[11], inp[29]);
        let s189 = veorq_u8(inp[14], inp[28]);
        let s190 = veorq_u8(s66, s95);
        let s191 = veorq_u8(inp[9], inp[22]);
        let s192 = veorq_u8(s70, s86);
        let s193 = veorq_u8(s72, s128);
        let s194 = veorq_u8(inp[39], s64);
        let s195 = veorq_u8(inp[41], s90);
        let s196 = veorq_u8(inp[7], s81);
        let s197 = veorq_u8(inp[56], s89);
        let s198 = veorq_u8(inp[20], inp[58]);
        let s199 = veorq_u8(inp[63], s92);
        let s200 = veorq_u8(inp[62], s90);
        let s201 = veorq_u8(inp[33], s126);
        let s202 = veorq_u8(inp[12], inp[26]);
        let s203 = veorq_u8(s79, s155);
        let s204 = veorq_u8(inp[17], inp[53]);
        let s205 = veorq_u8(s80, s93);
        let s206 = veorq_u8(inp[17], s71);
        let s207 = veorq_u8(inp[44], s101);
        let s208 = veorq_u8(inp[1], s89);
        let s209 = veorq_u8(inp[5], inp[36]);
        let s210 = veorq_u8(inp[41], inp[55]);
        let s211 = veorq_u8(s92, s98);
        let s212 = veorq_u8(s123, s130);
        let s213 = veorq_u8(inp[19], s74);
        let s214 = veorq_u8(inp[25], inp[49]);
        let s215 = veorq_u8(inp[37], s68);
        let s216 = veorq_u8(inp[15], s109);
        let s217 = veorq_u8(inp[8], inp[61]);
        let s218 = veorq_u8(inp[12], inp[48]);
        let s219 = veorq_u8(inp[31], s91);
        let s220 = veorq_u8(inp[3], s69);
        let s221 = veorq_u8(inp[11], s82);
        let s222 = veorq_u8(inp[28], s96);
        let s223 = veorq_u8(inp[14], inp[18]);
        let s224 = veorq_u8(inp[38], s119);
        let s225 = veorq_u8(inp[45], s89);
        let s226 = veorq_u8(s73, s85);
        let s227 = veorq_u8(s87, s134);
        let s228 = veorq_u8(s90, s135);
        let s229 = veorq_u8(inp[23], s104);
        let s230 = veorq_u8(inp[46], s156);
        let s231 = veorq_u8(s83, s183);
        let s232 = veorq_u8(s102, s148);
        let s233 = veorq_u8(inp[60], s105);
        let s234 = veorq_u8(inp[35], s73);
        let s235 = veorq_u8(inp[54], s94);
        let s236 = veorq_u8(s95, s169);
        let s237 = veorq_u8(inp[20], s140);
        let s238 = veorq_u8(s65, s72);
        let s239 = veorq_u8(inp[0], inp[47]);
        let s240 = veorq_u8(inp[8], s113);
        let s241 = veorq_u8(inp[44], s84);
        let s242 = veorq_u8(inp[30], s64);
        let s243 = veorq_u8(s103, s115);
        let s244 = veorq_u8(inp[24], inp[42]);
        let s245 = veorq_u8(s114, s129);
        let s246 = veorq_u8(inp[38], inp[47]);
        let s247 = veorq_u8(inp[6], s107);
        let s248 = veorq_u8(inp[30], s72);
        let s249 = veorq_u8(inp[34], s127);
        let s250 = veorq_u8(s82, s116);
        let s251 = veorq_u8(inp[33], s97);
        let s252 = veorq_u8(inp[2], inp[50]);
        let s253 = veorq_u8(inp[59], s101);
        let s254 = veorq_u8(inp[6], inp[43]);
        let s255 = veorq_u8(inp[48], s142);
        let s256 = veorq_u8(inp[7], s154);
        let s257 = veorq_u8(inp[44], s76);
        let s258 = veorq_u8(inp[3], s111);
        let s259 = veorq_u8(s70, s163);
        let s260 = veorq_u8(s106, s139);
        let s261 = veorq_u8(inp[13], s114);
        let s262 = veorq_u8(inp[36], s99);
        let s263 = veorq_u8(s100, s102);
        let s264 = veorq_u8(inp[56], s131);
        let s265 = veorq_u8(inp[52], s121);
        let s266 = veorq_u8(inp[37], s120);
        let s267 = veorq_u8(s76, s77);
        let s268 = veorq_u8(s78, s175);
        let s269 = veorq_u8(inp[21], inp[33]);
        let s270 = veorq_u8(inp[41], s117);
        let s271 = veorq_u8(inp[58], s122);
        let s272 = veorq_u8(inp[58], s95);
        let s273 = veorq_u8(s64, s110);
        let s274 = veorq_u8(s125, s133);
        let s275 = veorq_u8(inp[16], s105);
        let s276 = veorq_u8(s79, s81);
        let s277 = veorq_u8(inp[5], inp[45]);
        let s278 = veorq_u8(inp[4], inp[16]);
        let s279 = veorq_u8(s66, s87);
        let s280 = veorq_u8(inp[55], s218);
        let s281 = veorq_u8(s111, s172);
        let s282 = veorq_u8(inp[21], s97);
        let s283 = veorq_u8(inp[25], inp[30]);
        let s284 = veorq_u8(inp[9], s136);
        let s285 = veorq_u8(inp[31], s145);
        let s286 = veorq_u8(inp[7], inp[40]);
        let s287 = veorq_u8(inp[62], s179);
        let s288 = veorq_u8(s111, s178);
        let s289 = veorq_u8(s87, s137);
        let s290 = veorq_u8(s108, s166);
        let s291 = veorq_u8(inp[15], inp[57]);
        let s292 = veorq_u8(inp[1], inp[10]);
        let s293 = veorq_u8(s118, s147);
        let s294 = veorq_u8(s102, s149);
        let s295 = veorq_u8(inp[1], s151);
        let s296 = veorq_u8(inp[16], s160);
        let s297 = veorq_u8(inp[29], s80);
        let s298 = veorq_u8(s161, s206);
        let s299 = veorq_u8(s80, s210);
        let s300 = veorq_u8(s118, s129);
        let s301 = veorq_u8(inp[8], s152);
        let s302 = veorq_u8(s74, s106);
        let s303 = veorq_u8(s82, s86);
        let s304 = veorq_u8(s86, s167);
        let s305 = veorq_u8(s90, s92);
        let s306 = veorq_u8(inp[24], s117);
        let s307 = veorq_u8(inp[27], s72);
        let s308 = veorq_u8(inp[62], s83);
        let s309 = veorq_u8(inp[13], inp[32]);
        let s310 = veorq_u8(inp[53], s120);
        let s311 = veorq_u8(s92, s135);
        let s312 = veorq_u8(inp[36], inp[54]);
        let s313 = veorq_u8(inp[48], s91);
        let s314 = veorq_u8(inp[47], s119);
        let s315 = veorq_u8(inp[60], s100);
        let s316 = veorq_u8(inp[35], inp[46]);
        let s317 = veorq_u8(inp[61], s84);
        let s318 = veorq_u8(s84, s173);
        let s319 = veorq_u8(inp[26], inp[43]);
        let s320 = veorq_u8(inp[50], s97);
        let s321 = veorq_u8(inp[54], s134);
        let s322 = veorq_u8(s66, s89);
        let s323 = veorq_u8(inp[0], s101);
        let s324 = veorq_u8(inp[2], s124);
        let s325 = veorq_u8(inp[40], inp[49]);
        let s326 = veorq_u8(inp[59], s70);
        let s327 = veorq_u8(s71, s73);
        let s328 = veorq_u8(inp[10], inp[11]);
        let s329 = veorq_u8(inp[23], inp[28]);
        let s330 = veorq_u8(s109, s153);
        let s331 = veorq_u8(s114, s120);
        let s332 = veorq_u8(inp[18], inp[44]);
        let s333 = veorq_u8(inp[51], s83);
        let s334 = veorq_u8(inp[26], s74);
        let s335 = veorq_u8(inp[37], s160);
        let s336 = veorq_u8(inp[17], s93);
        let s337 = veorq_u8(inp[62], s103);
        let s338 = veorq_u8(s109, s197);
        let s339 = veorq_u8(inp[17], inp[18]);
        let s340 = veorq_u8(inp[33], s144);
        let s341 = veorq_u8(inp[14], inp[39]);
        let s342 = veorq_u8(s65, s133);
        let s343 = veorq_u8(s71, s116);
        let s344 = veorq_u8(inp[3], s108);
        let s345 = veorq_u8(inp[4], inp[50]);
        let s346 = veorq_u8(inp[51], s150);
        let s347 = veorq_u8(inp[19], inp[28]);
        let s348 = veorq_u8(s66, s91);
        let s349 = veorq_u8(inp[2], s67);
        let s350 = veorq_u8(inp[56], inp[58]);
        let s351 = veorq_u8(s69, s113);
        let s352 = veorq_u8(s72, s79);
        let s353 = veorq_u8(inp[29], s125);
        let s354 = veorq_u8(s68, s78);
        let s355 = veorq_u8(inp[27], s184);
        let s356 = veorq_u8(inp[42], s110);
        let s357 = veorq_u8(inp[22], s113);
        let s358 = veorq_u8(inp[41], s121);
        let s359 = veorq_u8(s81, s141);
        let s360 = veorq_u8(inp[59], s200);
        let s361 = veorq_u8(inp[20], s85);
        let s362 = veorq_u8(inp[2], s151);
        let s363 = veorq_u8(inp[26], inp[61]);
        let s364 = veorq_u8(inp[11], s93);
        let s365 = veorq_u8(s132, s220);
        let s366 = veorq_u8(s65, s101);
        let s367 = veorq_u8(inp[24], s129);
        let s368 = veorq_u8(inp[9], s69);
        let s369 = veorq_u8(inp[0], s187);
        let s370 = veorq_u8(inp[37], s94);
        let s371 = veorq_u8(s131, s157);
        let s372 = veorq_u8(inp[51], s99);
        let s373 = veorq_u8(s64, s201);
        let s374 = veorq_u8(s146, s165);
        let s375 = veorq_u8(s228, s374);
        let s376 = veorq_u8(s136, s229);
        let s377 = veorq_u8(inp[17], s70);
        let s378 = veorq_u8(s85, s147);
        let s379 = veorq_u8(s98, s181);
        let s380 = veorq_u8(s180, s182);
        let s381 = veorq_u8(s202, s230);
        let s382 = veorq_u8(s73, s149);
        let s383 = veorq_u8(s65, s122);
        let s384 = veorq_u8(s150, s157);
        let s385 = veorq_u8(s167, s383);
        let s386 = veorq_u8(s184, s207);
        let s387 = veorq_u8(inp[12], s64);
        let s388 = veorq_u8(s98, s236);
        let s389 = veorq_u8(s138, s234);
        let s390 = veorq_u8(inp[43], s185);
        let s391 = veorq_u8(s83, s139);
        let s392 = veorq_u8(s99, s292);
        let s393 = veorq_u8(inp[10], s239);
        let s394 = veorq_u8(s78, s159);
        let s395 = veorq_u8(s97, s127);
        let s396 = veorq_u8(s116, s241);
        let s397 = veorq_u8(inp[38], inp[44]);
        let s398 = veorq_u8(s80, s117);
        let s399 = veorq_u8(s128, s242);
        let s400 = veorq_u8(inp[31], s103);
        let s401 = veorq_u8(s170, s244);
        let s402 = veorq_u8(inp[14], s296);
        let s403 = veorq_u8(s68, s209);
        let s404 = veorq_u8(s85, s123);
        let s405 = veorq_u8(inp[8], s298);
        let s406 = veorq_u8(s88, s103);
        let s407 = veorq_u8(inp[1], s162);
        let s408 = veorq_u8(inp[38], inp[39]);
        let s409 = veorq_u8(inp[6], s203);
        let s410 = veorq_u8(s189, s303);
        let s411 = veorq_u8(inp[30], s141);
        let s412 = veorq_u8(inp[54], s253);
        let s413 = veorq_u8(s100, s190);
        let s414 = veorq_u8(s110, s162);
        let s415 = veorq_u8(inp[10], s171);
        let s416 = veorq_u8(s126, s143);
        let s417 = veorq_u8(inp[2], s193);
        let s418 = veorq_u8(inp[11], s191);
        let s419 = veorq_u8(inp[21], s161);
        let s420 = veorq_u8(inp[26], s212);
        let s421 = veorq_u8(s105, s124);
        let s422 = veorq_u8(s115, s194);
        let s423 = veorq_u8(s257, s311);
        let s424 = veorq_u8(inp[7], s312);
        let s425 = veorq_u8(s148, s259);
        let s426 = veorq_u8(inp[40], inp[57]);
        let s427 = veorq_u8(inp[7], s142);
        let s428 = veorq_u8(inp[11], s75);
        let s429 = veorq_u8(inp[60], s321);
        let s430 = veorq_u8(s156, s322);
        let s431 = veorq_u8(inp[30], inp[36]);
        let s432 = veorq_u8(s78, s171);
        let s433 = veorq_u8(s79, s174);
        let s434 = veorq_u8(inp[15], inp[45]);
        let s435 = veorq_u8(s96, s323);
        let s436 = veorq_u8(s112, s118);
        let s437 = veorq_u8(s79, s145);
        let s438 = veorq_u8(s123, s327);
        let s439 = veorq_u8(s162, s325);
        let s440 = veorq_u8(inp[36], s132);
        let s441 = veorq_u8(s78, s112);
        let s442 = veorq_u8(s264, s291);
        let s443 = veorq_u8(inp[3], s122);
        let s444 = veorq_u8(inp[38], s265);
        let s445 = veorq_u8(s140, s166);
        let s446 = veorq_u8(s174, s195);
        let s447 = veorq_u8(s110, s196);
        let s448 = veorq_u8(s125, s163);
        let s449 = veorq_u8(inp[56], s334);
        let s450 = veorq_u8(inp[12], inp[29]);
        let s451 = veorq_u8(s249, s270);
        let s452 = veorq_u8(inp[5], s130);
        let s453 = veorq_u8(inp[39], s170);
        let s454 = veorq_u8(inp[59], s114);
        let s455 = veorq_u8(inp[29], s136);
        let s456 = veorq_u8(inp[59], s194);
        let s457 = veorq_u8(s86, s316);
        let s458 = veorq_u8(inp[13], s152);
        let s459 = veorq_u8(inp[48], s164);
        let s460 = veorq_u8(s94, s120);
        let s461 = veorq_u8(s152, s168);
        let s462 = veorq_u8(s177, s262);
        let s463 = veorq_u8(inp[4], s131);
        let s464 = veorq_u8(s104, s215);
        let s465 = veorq_u8(s219, s226);
        let s466 = veorq_u8(s121, s242);
        let s467 = veorq_u8(inp[12], s89);
        let s468 = veorq_u8(s176, s214);
        let s469 = veorq_u8(inp[27], inp[47]);
        let s470 = veorq_u8(inp[18], inp[53]);
        let s471 = veorq_u8(inp[42], s105);
        let s472 = veorq_u8(s201, s470);
        let s473 = veorq_u8(inp[33], s69);
        let s474 = veorq_u8(s68, s159);
        let s475 = veorq_u8(s69, s70);
        let s476 = veorq_u8(inp[31], inp[47]);
        let s477 = veorq_u8(inp[57], s263);
        let s478 = veorq_u8(s81, s178);
        let s479 = veorq_u8(s164, s347);
        let s480 = veorq_u8(inp[55], s75);
        let s481 = veorq_u8(s105, s108);
        let s482 = veorq_u8(s131, s227);
        let s483 = veorq_u8(s148, s252);
        let s484 = veorq_u8(inp[15], inp[32]);
        let s485 = veorq_u8(s81, s217);
        let s486 = veorq_u8(s106, s222);
        let s487 = veorq_u8(inp[24], inp[45]);
        let s488 = veorq_u8(inp[28], inp[62]);
        let s489 = veorq_u8(inp[32], s138);
        let s490 = veorq_u8(s80, s487);
        let s491 = veorq_u8(s137, s211);
        let s492 = veorq_u8(s91, s198);
        let s493 = veorq_u8(inp[16], s130);
        let s494 = veorq_u8(s145, s192);
        let s495 = veorq_u8(inp[5], inp[13]);
        let s496 = veorq_u8(inp[21], s231);
        let s497 = veorq_u8(s132, s250);
        let s498 = veorq_u8(inp[4], s70);
        let s499 = veorq_u8(inp[41], s116);
        let s500 = veorq_u8(inp[57], s140);
        let s501 = veorq_u8(s98, s254);
        let s502 = veorq_u8(s87, s350);
        let s503 = veorq_u8(s164, s318);
        let s504 = veorq_u8(s77, s91);
        let s505 = veorq_u8(s143, s245);
        let s506 = veorq_u8(inp[8], s188);
        let s507 = veorq_u8(inp[60], s145);
        let s508 = veorq_u8(s75, s147);
        let s509 = veorq_u8(s132, s230);
        let s510 = veorq_u8(inp[26], s88);
        let s511 = veorq_u8(inp[46], s124);
        let s512 = veorq_u8(s76, s102);
        let s513 = veorq_u8(s158, s258);
        let s514 = veorq_u8(inp[40], s150);
        let s515 = veorq_u8(inp[61], s73);
        let s516 = veorq_u8(s279, s514);
        let s517 = veorq_u8(s71, s175);
        let s518 = veorq_u8(inp[9], s168);
        let s519 = veorq_u8(s266, s355);
        let s520 = veorq_u8(inp[34], s221);
        let s521 = veorq_u8(inp[63], s68);
        let s522 = veorq_u8(s117, s356);
        let s523 = veorq_u8(s314, s520);
        let s524 = veorq_u8(s186, s224);
        let s525 = veorq_u8(s124, s141);
        let s526 = veorq_u8(inp[61], s118);
        let s527 = veorq_u8(s100, s124);
        let s528 = veorq_u8(s99, s297);
        let s529 = veorq_u8(inp[25], s144);
        let s530 = veorq_u8(s109, s154);
        let s531 = veorq_u8(inp[19], s364);
        let s532 = veorq_u8(s95, s182);
        let s533 = veorq_u8(s106, s165);
        let s534 = veorq_u8(inp[14], s120);
        let s535 = veorq_u8(inp[26], s153);
        let s536 = veorq_u8(s89, s158);
        let s537 = veorq_u8(s108, s122);
        let s538 = veorq_u8(s209, s535);
        let s539 = veorq_u8(s139, s271);
        let s540 = veorq_u8(inp[18], inp[19]);
        let s541 = veorq_u8(s111, s128);
        let s542 = veorq_u8(s138, s541);
        let s543 = veorq_u8(inp[62], s199);
        let s544 = veorq_u8(s95, s365);
        let s545 = veorq_u8(s133, s149);
        let s546 = veorq_u8(inp[39], s233);
        let s547 = veorq_u8(s97, s363);
        let s548 = veorq_u8(inp[6], inp[30]);
        let s549 = veorq_u8(inp[23], s366);
        let s550 = veorq_u8(inp[24], inp[49]);
        let s551 = veorq_u8(s88, s153);
        let s552 = veorq_u8(inp[9], s267);
        let s553 = veorq_u8(inp[20], s112);
        let s554 = veorq_u8(inp[32], s142);
        let s555 = veorq_u8(s69, s152);
        let s556 = veorq_u8(s86, s269);
        let s557 = veorq_u8(s151, s368);
        let s558 = veorq_u8(inp[20], s119);
        let s559 = veorq_u8(s74, s90);
        let s560 = veorq_u8(s121, s559);
        let s561 = veorq_u8(s76, s137);
        let s562 = veorq_u8(s143, s208);
        let s563 = veorq_u8(s260, s282);
        let s564 = veorq_u8(s123, s143);
        let s565 = veorq_u8(s93, s108);
        let s566 = veorq_u8(inp[54], s103);
        let s567 = veorq_u8(s176, s223);
        let s568 = veorq_u8(s136, s138);
        let s569 = veorq_u8(inp[51], s98);
        let s570 = veorq_u8(s75, s372);
        let s571 = veorq_u8(inp[57], s284);
        let s572 = veorq_u8(inp[18], s137);
        let s573 = veorq_u8(s88, s188);
        let s574 = veorq_u8(s95, s177);
        let s575 = veorq_u8(s67, s127);
        let s576 = veorq_u8(s222, s277);
        let s577 = veorq_u8(inp[24], s225);
        let s578 = veorq_u8(s268, s324);
        out[0] = veor3q_u8(veor3q_u8(s226, s227, s285), s373, s375);
        out[1] = veorq_u8(veor3q_u8(veor3q_u8(s110, s121, s286), s287, s288), s376);
        out[2] = veor3q_u8(
            veor3q_u8(veor3q_u8(s203, s289, s377), s378, s379),
            s380,
            s381,
        );
        out[3] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[14], inp[27], s204), s205, s231),
                s232,
                s290,
            ),
            s382,
        );
        out[4] = veorq_u8(veor3q_u8(veor3q_u8(s206, s233, s291), s384, s385), s386);
        out[5] = veor3q_u8(
            veor3q_u8(veor3q_u8(veor3q_u8(s79, s90, s158), s168, s185), s235, s387),
            s388,
            s389,
        );
        out[6] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s84, s159, s165), s186, s237),
                s238,
                s390,
            ),
            s391,
            s392,
        );
        out[7] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[54], inp[59], s240), s285, s393),
                s394,
                s395,
            ),
            s396,
        );
        out[8] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s187, s237, s243), s286, s293),
                s397,
                s398,
            ),
            s399,
        );
        out[9] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[35], s185, s207), s208, s229),
                s245,
                s294,
            ),
            s400,
            s401,
        );
        out[10] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[35], s246, s295), s297, s375),
                s402,
                s403,
            ),
            s404,
        );
        out[11] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[28], s171, s232), s247, s248),
                s249,
                s405,
            ),
            s406,
        );
        out[12] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(veor3q_u8(s76, s87, s126), s188, s250), s299, s300),
                s382,
                s407,
            ),
            s408,
        );
        out[13] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s85, s211, s251), s299, s301),
                s302,
                s409,
            ),
            s410,
        );
        out[14] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s84, s157, s252), s304, s305),
                s411,
                s412,
            ),
            s413,
            s414,
        );
        out[15] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[53], s119, s185), s254, s255),
                    s306,
                    s307,
                ),
                s308,
                s415,
            ),
            s416,
        );
        out[16] = veor3q_u8(
            veor3q_u8(veor3q_u8(s192, s253, s256), s417, s418),
            s419,
            s420,
        );
        out[17] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[11], inp[22], s172), s309, s310),
                s421,
                s422,
            ),
            s423,
        );
        out[18] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[3], inp[32], s105), s114, s213),
                s234,
                s313,
            ),
            s423,
            s424,
        );
        out[19] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s73, s101, s202), s214, s258),
                s260,
                s314,
            ),
            s315,
            s425,
        );
        out[20] = veor3q_u8(
            veor3q_u8(veor3q_u8(veor3q_u8(s72, s93, s96), s179, s303), s316, s317),
            s422,
            s426,
        );
        out[21] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s118, s134, s215), s261, s306),
                s318,
                s390,
            ),
            s407,
            s427,
        );
        out[22] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s77, s138, s301), s319, s320),
                s428,
                s429,
            ),
            s430,
        );
        out[23] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[4], s82, s126), s161, s308),
                s378,
                s431,
            ),
            s432,
            s433,
        );
        out[24] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s110, s135, s262), s263, s324),
                s434,
                s435,
            ),
            s436,
        );
        out[25] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s89, s109, s210), s213, s293),
                s326,
                s437,
            ),
            s438,
            s439,
        );
        out[26] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[62], s102, s141), s293, s328),
                s329,
                s440,
            ),
            s441,
            s442,
        );
        out[27] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[33], inp[51], s202), s289, s330),
                s331,
                s421,
            ),
            s443,
            s444,
        );
        out[28] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[42], s266, s304), s328, s332),
                s333,
                s438,
            ),
            s445,
            s446,
        );
        out[29] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[8], inp[25], inp[35]), s102, s159),
                    s299,
                    s309,
                ),
                s426,
                s447,
            ),
            s448,
        );
        out[30] = veor3q_u8(
            veor3q_u8(veor3q_u8(s127, s179, s239), s255, s267),
            s335,
            s449,
        );
        out[31] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[25], s176, s236), s237, s268),
                s336,
                s337,
            ),
            s338,
            s414,
        );
        out[32] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[6], s67, s123), s141, s215),
                s269,
                s339,
            ),
            s450,
            s451,
        );
        out[33] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(veor3q_u8(s67, s90, s132), s216, s240), s246, s326),
                s340,
                s452,
            ),
            s453,
        );
        out[34] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[1], s84, s147), s255, s341),
                s342,
                s376,
            ),
            s451,
            s454,
        );
        out[35] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s196, s248, s271), s287, s343),
                s455,
                s456,
            ),
            s457,
        );
        out[36] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s143, s191, s197), s217, s294),
                s435,
                s458,
            ),
            s459,
        );
        out[37] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s74, s218, s251), s272, s292),
                s398,
                s460,
            ),
            s461,
            s462,
        );
        out[38] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[0], s102, s214), s236, s273),
                s274,
                s310,
            ),
            s319,
        );
        out[39] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s135, s142, s160), s216, s344),
                s463,
                s464,
            ),
            s465,
        );
        out[40] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[7], inp[16], s83), s129, s173),
                    s195,
                    s220,
                ),
                s221,
                s257,
            ),
            s413,
            s466,
        );
        out[41] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[23], inp[32], s156), s182, s275),
            s467,
            s468,
        );
        out[42] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[17], inp[52], s105), s107, s109),
                    s165,
                    s178,
                ),
                s221,
                s345,
            ),
            s469,
        );
        out[43] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[8], inp[40], inp[63]), s89, s93),
                    s101,
                    s116,
                ),
                s220,
                s235,
            ),
            s471,
            s472,
        );
        out[44] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[26], s67, s170), s183, s264),
                s276,
                s471,
            ),
            s473,
            s474,
        );
        out[45] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[29], inp[49], s84), s96, s104),
                    s182,
                    s256,
                ),
                s277,
                s300,
            ),
            s346,
            s475,
        );
        out[46] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s125, s204, s235), s344, s476),
                s477,
                s478,
            ),
            s479,
        );
        out[47] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[12], s145, s189), s198, s249),
                s480,
                s481,
            ),
            s482,
            s483,
        );
        out[48] = veorq_u8(veor3q_u8(veor3q_u8(inp[17], s172, s247), s484, s485), s486);
        out[49] = veorq_u8(veor3q_u8(veor3q_u8(s256, s339, s488), s489, s490), s491);
        out[50] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[36], s173, s238), s302, s305),
            s402,
            s492,
        );
        out[51] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[39], inp[56], s199), s213, s258),
                s384,
                s406,
            ),
            s493,
            s494,
        );
        out[52] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[58], s64, s128), s348, s427),
                s444,
                s495,
            ),
            s496,
            s497,
        );
        out[53] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[11], s196, s342), s349, s490),
                s498,
                s499,
            ),
            s500,
        );
        out[54] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[5], inp[10], inp[48]), s71, s82),
                    s163,
                    s176,
                ),
                s198,
                s439,
            ),
            s485,
            s501,
        );
        out[55] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[29], s105, s112), s210, s223),
                s240,
                s266,
            ),
            s502,
            s503,
        );
        out[56] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[35], s66, s79), s107, s144),
                    s147,
                    s237,
                ),
                s306,
                s351,
            ),
            s418,
        );
        out[57] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[15], inp[42], s82), s107, s153),
                    s278,
                    s347,
                ),
                s352,
                s504,
            ),
            s505,
        );
        out[58] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[34], inp[43], s71), s191, s203),
                    s214,
                    s270,
                ),
                s307,
                s389,
            ),
            s456,
        );
        out[59] = veor3q_u8(
            veor3q_u8(veor3q_u8(veor3q_u8(s66, s77, s80), s92, s165), s193, s420),
            s483,
            s506,
        );
        out[60] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[13], s177, s194), s472, s507),
            s508,
            s509,
        );
        out[61] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[16], s102, s118), s136, s205),
                    s207,
                    s269,
                ),
                s277,
                s313,
            ),
            s399,
            s510,
        );
        out[62] = veor3q_u8(
            veor3q_u8(veor3q_u8(s144, s170, s325), s391, s511),
            s512,
            s513,
        );
        out[63] = veorq_u8(veor3q_u8(veor3q_u8(s185, s332, s353), s373, s515), s516);
        out[64] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[36], s134, s228), s275, s280),
                s332,
                s425,
            ),
            s517,
        );
        out[65] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[5], s124, s155), s180, s205),
                    s261,
                    s321,
                ),
                s344,
                s354,
            ),
            s410,
            s480,
        );
        out[66] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[11], inp[28], s107), s110, s132),
                    s148,
                    s226,
                ),
                s516,
                s518,
            ),
            s519,
        );
        out[67] = veor3q_u8(veor3q_u8(veor3q_u8(s65, s81, s405), s431, s521), s522, s523);
        out[68] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[5], inp[33], inp[50]), s162, s260),
                    s263,
                    s357,
                ),
                s358,
                s359,
            ),
            s524,
        );
        out[69] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[63], s142, s196), s340, s360),
                s492,
                s508,
            ),
            s525,
        );
        out[70] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[19], inp[41], inp[42]), s104, s171),
                    s290,
                    s323,
                ),
                s450,
                s526,
            ),
            s527,
        );
        out[71] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[52], s88, s106), s157, s197),
                    s253,
                    s296,
                ),
                s361,
                s473,
            ),
            s499,
            s528,
        );
        out[72] = veor3q_u8(
            veor3q_u8(veor3q_u8(veor3q_u8(s92, s99, s130), s307, s322), s362, s363),
            s529,
            s530,
        );
        out[73] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[15], s94, s146), s173, s193),
                s280,
                s411,
            ),
            s474,
            s531,
        );
        out[74] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[3], inp[5], inp[42]), s305, s335),
                s364,
                s395,
            ),
            s404,
            s532,
        );
        out[75] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[18], inp[37], s86), s101, s140),
                    s161,
                    s330,
                ),
                s465,
                s501,
            ),
            s533,
        );
        out[76] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[12], inp[31], s66), s71, s76),
                    s112,
                    s158,
                ),
                s228,
                s359,
            ),
            s498,
            s534,
        );
        out[77] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s100, s106, s126), s164, s360),
                s536,
                s537,
            ),
            s538,
        );
        out[78] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[9], inp[31], s85), s101, s162),
                    s286,
                    s346,
                ),
                s416,
                s459,
            ),
            s539,
        );
        out[79] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s68, s144, s153), s203, s219),
                s279,
                s540,
            ),
            s542,
        );
        out[80] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[52], s113, s152), s181, s232),
                s278,
                s448,
            ),
            s543,
            s544,
        );
        out[81] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[57], s89, s139), s174, s180),
                    s238,
                    s281,
                ),
                s300,
                s343,
            ),
            s545,
        );
        out[82] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[22], s83, s109), s335, s462),
            s546,
            s547,
        );
        out[83] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[10], inp[52], s154), s204, s315),
                s457,
                s538,
            ),
            s542,
            s548,
        );
        out[84] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[18], inp[43], inp[58]), s85, s118),
                s155,
                s244,
            ),
            s337,
            s477,
        );
        out[85] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[7], s86, s149), s268, s295),
                s509,
                s549,
            ),
            s550,
            s551,
        );
        out[86] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[5], s75, s102), s190, s339),
                s367,
                s503,
            ),
            s536,
            s552,
        );
        out[87] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[0], inp[35], s91), s103, s113),
                    s137,
                    s154,
                ),
                s200,
                s233,
            ),
            s341,
            s519,
        );
        out[88] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[37], s141, s205), s280, s333),
                s351,
                s461,
            ),
            s549,
        );
        out[89] = veor3q_u8(
            veor3q_u8(veor3q_u8(veor3q_u8(s94, s99, s150), s310, s517), s545, s553),
            s554,
            s555,
        );
        out[90] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s149, s181, s229), s281, s331),
                s368,
                s415,
            ),
            s510,
        );
        out[91] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[32], s67, s77), s78, s93),
                    s143,
                    s250,
                ),
                s274,
                s409,
            ),
            s476,
            s527,
        );
        out[92] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(s117, s128, s183), s186, s191),
                    s243,
                    s278,
                ),
                s369,
                s424,
            ),
            s532,
        );
        out[93] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[15], inp[28], inp[53]), inp[56], s81),
                    s166,
                    s198,
                ),
                s320,
                s466,
            ),
            s497,
            s526,
        );
        out[94] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[1], s72, s181), s198, s200),
                s346,
                s486,
            ),
            s556,
        );
        out[95] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s94, s114, s212), s240, s274),
                s345,
                s348,
            ),
            s352,
            s557,
        );
        out[96] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[42], inp[43], s167), s273, s369),
                s381,
                s428,
            ),
            s558,
            s560,
        );
        out[97] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[22], inp[58], s111), s125, s140),
                    s248,
                    s251,
                ),
                s345,
                s370,
            ),
            s561,
        );
        out[98] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[35], s116, s205), s289, s365),
                s489,
                s537,
            ),
            s546,
        );
        out[99] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s169, s180, s189), s352, s362),
                s408,
                s562,
            ),
            s563,
        );
        out[100] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[30], inp[44], s276), s354, s361),
                s385,
                s430,
            ),
            s506,
        );
        out[101] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[4], inp[29], inp[39]), s113, s298),
                    s317,
                    s356,
                ),
                s429,
                s434,
            ),
            s525,
        );
        out[102] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(veor3q_u8(s81, s86, s113), s150, s187), s225, s246),
                s336,
                s513,
            ),
            s564,
        );
        out[103] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[33], inp[63], s85), s119, s146),
                    s273,
                    s326,
                ),
                s449,
                s455,
            ),
            s478,
        );
        out[104] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[21], s177, s195), s196, s199),
                    s247,
                    s304,
                ),
                s313,
                s361,
            ),
            s452,
            s565,
        );
        out[105] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[27], s85, s222), s224, s275),
                s367,
                s371,
            ),
            s529,
            s566,
        );
        out[106] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[5], inp[34], inp[60]), s80, s126),
                    s142,
                    s331,
                ),
                s387,
                s394,
            ),
            s469,
            s539,
        );
        out[107] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[32], inp[34], inp[46]), inp[51], s103),
                s279,
                s320,
            ),
            s552,
            s560,
        );
        out[108] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(
                        veor3q_u8(veor3q_u8(inp[0], inp[34], inp[55]), s93, s121),
                        s208,
                        s219,
                    ),
                    s281,
                    s283,
                ),
                s340,
                s342,
            ),
            s433,
        );
        out[109] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[20], s88, s91), s143, s170),
                    s211,
                    s224,
                ),
                s248,
                s454,
            ),
            s530,
            s567,
        );
        out[110] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s107, s123, s225), s232, s241),
                s282,
                s446,
            ),
            s491,
        );
        out[111] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[15], inp[38], s83), s86, s176),
                    s180,
                    s301,
                ),
                s327,
                s521,
            ),
            s544,
            s553,
        );
        out[112] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[4], s211, s223), s256, s283),
                s290,
                s357,
            ),
            s412,
            s568,
        );
        out[113] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[35], s206, s272), s274, s284),
                s329,
                s523,
            ),
            s551,
            s569,
        );
        out[114] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[23], s98, s103), s106, s127),
                    s134,
                    s204,
                ),
                s288,
                s362,
            ),
            s403,
        );
        out[115] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[10], inp[14], s106), s133, s174),
                s245,
                s370,
            ),
            s507,
        );
        out[116] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[46], s73, s93), s106, s107),
                    s169,
                    s244,
                ),
                s371,
                s417,
            ),
            s460,
        );
        out[117] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[25], inp[53], s127), s129, s189),
                    s217,
                    s276,
                ),
                s287,
                s358,
            ),
            s442,
            s531,
        );
        out[118] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[27], s79, s119), s262, s453),
                s543,
                s547,
            ),
            s561,
        );
        out[119] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s80, s167, s224), s319, s354),
                s458,
                s502,
            ),
            s570,
        );
        out[120] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[3], inp[24], s71), s126, s153),
                    s223,
                    s317,
                ),
                s358,
                s388,
            ),
            s467,
            s571,
        );
        out[121] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[46], s75, s79), s111, s130),
                    s150,
                    s201,
                ),
                s247,
                s261,
            ),
            s285,
            s571,
        );
        out[122] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s74, s140, s193), s264, s265),
                s515,
                s566,
            ),
            s568,
        );
        out[123] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[3], inp[13], s84), s102, s117),
                    s144,
                    s195,
                ),
                s357,
                s380,
            ),
            s572,
        );
        out[124] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[21], inp[27], inp[45]), inp[60], s160),
                    s207,
                    s212,
                ),
                s221,
                s311,
            ),
            s359,
            s558,
        );
        out[125] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[1], inp[50], s104), s130, s158),
                    s192,
                    s204,
                ),
                s217,
                s272,
            ),
            s330,
            s337,
        );
        out[126] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[6], inp[60], inp[63]), s106, s135),
                    s184,
                    s238,
                ),
                s239,
                s512,
            ),
            s534,
        );
        out[127] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(
                        veor3q_u8(veor3q_u8(inp[39], inp[44], inp[59]), s108, s110),
                        s133,
                        s144,
                    ),
                    s191,
                    s225,
                ),
                s329,
                s333,
            ),
            s468,
        );
        out[128] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(veor3q_u8(s67, s74, s92), s104, s164), s166, s338),
                s370,
                s570,
            ),
            s573,
        );
        out[129] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[16], inp[55], s108), s124, s125),
                    s246,
                    s259,
                ),
                s338,
                s436,
            ),
            s574,
        );
        out[130] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[2], inp[34], inp[40]), s70, s115),
                    s181,
                    s190,
                ),
                s199,
                s386,
            ),
            s495,
        );
        out[131] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[51], s108, s138), s154, s193),
                s197,
                s231,
            ),
            s397,
            s575,
        );
        out[132] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[3], inp[56], s134), s155, s158),
                    s173,
                    s278,
                ),
                s528,
                s564,
            ),
            s576,
        );
        out[133] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(
                        veor3q_u8(veor3q_u8(inp[34], inp[41], inp[62]), s69, s125),
                        s168,
                        s186,
                    ),
                    s216,
                    s234,
                ),
                s257,
                s348,
            ),
            s565,
        );
        out[134] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[41], inp[60], s81), s87, s92),
                    s127,
                    s139,
                ),
                s261,
                s273,
            ),
            s295,
            s488,
        );
        out[135] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[6], inp[22], inp[40]), s80, s103),
                    s149,
                    s151,
                ),
                s159,
                s177,
            ),
            s419,
            s463,
        );
        out[136] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(
                        veor3q_u8(veor3q_u8(inp[11], inp[20], inp[47]), inp[63], s71),
                        s77,
                        s133,
                    ),
                    s186,
                    s202,
                ),
                s252,
                s351,
            ),
            s371,
            s548,
        );
        out[137] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[52], s93, s159), s294, s367),
                s396,
                s484,
            ),
            s563,
        );
        out[138] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[4], inp[63], s157), s179, s209),
                    s227,
                    s259,
                ),
                s328,
                s343,
            ),
            s443,
            s567,
        );
        out[139] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[32], s108, s151), s190, s195),
                    s219,
                    s271,
                ),
                s302,
                s379,
            ),
            s518,
        );
        out[140] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[49], s113, s132), s280, s493),
            s522,
            s556,
        );
        out[141] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[5], inp[14], s94), s101, s119),
                    s141,
                    s156,
                ),
                s272,
                s500,
            ),
            s573,
        );
        out[142] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[9], s99, s243), s283, s355),
            s432,
            s540,
        );
        out[143] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[0], inp[58], s117), s139, s213),
                s267,
                s282,
            ),
            s401,
            s555,
        );
        out[144] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[21], inp[37], s73), s90, s187),
                    s194,
                    s254,
                ),
                s276,
                s441,
            ),
            s554,
            s562,
        );
        out[145] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[22], inp[53], s87), s118, s122),
                    s131,
                    s168,
                ),
                s334,
                s341,
            ),
            s475,
            s577,
        );
        out[146] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[3], s75, s148), s175, s208),
                    s251,
                    s353,
                ),
                s377,
                s440,
            ),
            s445,
            s575,
        );
        out[147] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[5], s122, s135), s284, s393),
                s494,
                s504,
            ),
            s533,
            s550,
        );
        out[148] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[4], inp[12], inp[27]), s116, s163),
                    s215,
                    s216,
                ),
                s372,
                s400,
            ),
            s572,
            s574,
        );
        out[149] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[13], inp[37], inp[52]), inp[63], s190),
                    s241,
                    s353,
                ),
                s447,
                s479,
            ),
            s482,
        );
        out[150] = veor3q_u8(
            veor3q_u8(veor3q_u8(inp[4], inp[9], s68), s281, s505),
            s576,
            s578,
        );
        out[151] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s137, s171, s172), s216, s255),
                s350,
                s366,
            ),
            s481,
            s569,
        );
        out[152] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[12], inp[61], s161), s192, s200),
                    s235,
                    s265,
                ),
                s275,
                s283,
            ),
            s511,
        );
        out[153] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[8], inp[13], s71), s199, s312),
                s349,
                s437,
            ),
            s464,
            s524,
        );
        out[154] = veorq_u8(
            veor3q_u8(
                veor3q_u8(
                    veor3q_u8(veor3q_u8(inp[12], s103, s122), s147, s169),
                    s212,
                    s213,
                ),
                s315,
                s336,
            ),
            s392,
        );
        out[155] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(s112, s138, s189), s243, s309),
                s360,
                s577,
            ),
            s578,
        );
        out[156] = vdupq_n_u8(0);
        out[157] = veor3q_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[5], inp[15], inp[47]), s144, s174),
                s192,
                s288,
            ),
            s308,
            s349,
        );
        out[158] = veorq_u8(
            veor3q_u8(
                veor3q_u8(veor3q_u8(inp[39], inp[40], s75), s188, s270),
                s369,
                s496,
            ),
            s557,
        );
        out[159] = vdupq_n_u8(0);
    }
}

#[inline(always)]
unsafe fn product_bs(
    af: &[uint8x16_t; 160],
    bf: &[uint8x16_t; 160],
    ax: &[uint8x16_t; 64],
    bx: &[uint8x16_t; 64],
    out: &mut [uint8x16_t; 160],
) {
    unsafe {
        for p in 0..64 {
            out[p] = veorq_u8(vandq_u8(af[p], bx[p]), vandq_u8(ax[p], bf[p]));
        }
        for p in 0..64 {
            out[64 + p] = veorq_u8(
                veorq_u8(vandq_u8(af[64 + p], bx[p]), vandq_u8(af[p], bf[p])),
                vandq_u8(ax[p], bf[64 + p]),
            );
        }
        for p in 0..32 {
            out[128 + p] = veorq_u8(
                veorq_u8(vandq_u8(af[128 + p], bx[p]), vandq_u8(af[64 + p], bf[p])),
                veorq_u8(vandq_u8(af[p], bf[64 + p]), vandq_u8(ax[p], bf[128 + p])),
            );
        }
    }
}

#[inline(always)]
unsafe fn fold_bs(prod: &[uint8x16_t; 160], eq: F128, res: &mut [F128; 160]) {
    unsafe {
        for j in 0..160 {
            let pf = vreinterpretq_u64_u8(prod[j]);
            res[j] += eq
                * F128 {
                    lo: vgetq_lane_u64::<0>(pf),
                    hi: vgetq_lane_u64::<1>(pf),
                };
        }
    }
}

#[inline]
unsafe fn transpose16x16(r: &mut [uint8x16_t; 16]) {
    unsafe {
        for i in 0..16 {
            if i & 1 == 0 {
                let (x, y) = (r[i], r[i + 1]);
                r[i] = vtrn1q_u8(x, y);
                r[i + 1] = vtrn2q_u8(x, y);
            }
        }
        for i in 0..16 {
            if i & 2 == 0 {
                let x = vreinterpretq_u16_u8(r[i]);
                let y = vreinterpretq_u16_u8(r[i + 2]);
                r[i] = vreinterpretq_u8_u16(vtrn1q_u16(x, y));
                r[i + 2] = vreinterpretq_u8_u16(vtrn2q_u16(x, y));
            }
        }
        for i in 0..16 {
            if i & 4 == 0 {
                let x = vreinterpretq_u32_u8(r[i]);
                let y = vreinterpretq_u32_u8(r[i + 4]);
                r[i] = vreinterpretq_u8_u32(vtrn1q_u32(x, y));
                r[i + 4] = vreinterpretq_u8_u32(vtrn2q_u32(x, y));
            }
        }
        for i in 0..16 {
            if i & 8 == 0 {
                let x = vreinterpretq_u64_u8(r[i]);
                let y = vreinterpretq_u64_u8(r[i + 8]);
                r[i] = vreinterpretq_u8_u64(vtrn1q_u64(x, y));
                r[i + 8] = vreinterpretq_u8_u64(vtrn2q_u64(x, y));
            }
        }
    }
}

/// Like [`transpose_128x128`] but reads each 16-byte row as `lo`'s 8 bytes at
/// `base_lo + row*8` (low lanes) ‖ `hi`'s 8 bytes at `base_hi + row*8` (high
/// lanes) — straight from the packed witnesses, no intermediate interleave buf.
/// So `dst[0..64]` are `lo`'s planes and `dst[64..128]` are `hi`'s. (`lo == hi`
/// with different bases pairs two blocks of one witness.)
fn transpose_128x128_2src(
    lo: &[u8],
    base_lo: usize,
    hi: &[u8],
    base_hi: usize,
    dst: &mut [uint8x16_t; 128],
) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                let row = gi * 8 + k;
                let l = vld1_u8(lo.as_ptr().add(base_lo + row * 8));
                let h = vld1_u8(hi.as_ptr().add(base_hi + row * 8));
                q[k] = vcombine_u8(l, h);
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                dst[c * 8 + k] = v[c];
            }
        }
    }
}

fn transpose_128x128(src: &[u8], dst: &mut [uint8x16_t; 128]) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let so = gi * 8 * 16;
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                q[k] = vld1q_u8(src.as_ptr().add(so + k * 16));
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                dst[c * 8 + k] = v[c];
            }
        }
    }
}

#[inline(always)]
unsafe fn fold_c(cp: &[uint8x16_t; 64], eq: F128, wbar: &mut [F128; 64]) {
    unsafe {
        for k in 0..64 {
            let pf = vreinterpretq_u64_u8(cp[k]);
            wbar[k] += eq
                * F128 {
                    lo: vgetq_lane_u64::<0>(pf),
                    hi: vgetq_lane_u64::<1>(pf),
                };
        }
    }
}

/// NEON-resident unreduced accumulator for one coordinate: the three Karatsuba
/// parts of `Σ eq·x` kept as vectors — `[ll, cross, hh]` where `ll = Σ lo·lo`,
/// `cross = Σ (lo·hi ^ hi·lo)`, `hh = Σ hi·hi`. Folded to (r0..r3) + reduced
/// mod p ONCE per chunk. All-vector: no per-mult lane extracts, no reduction.
type UnredAcc = [uint64x2_t; 3];

#[inline(always)]
unsafe fn pmull_u64(a: u64, b: u64) -> uint64x2_t {
    unsafe { vreinterpretq_u64_p128(vmull_p64(a, b)) }
}

/// `acc ^= eq · x` unreduced: 4 PMULL + 4 VEOR, entirely in NEON registers.
#[inline(always)]
unsafe fn mul_acc_unred(acc: &mut UnredAcc, eq: F128, x: uint64x2_t) {
    unsafe {
        let xl = vgetq_lane_u64::<0>(x);
        let xh = vgetq_lane_u64::<1>(x);
        acc[0] = veorq_u64(acc[0], pmull_u64(eq.lo, xl));
        acc[1] = veorq_u64(
            acc[1],
            veorq_u64(pmull_u64(eq.lo, xh), pmull_u64(eq.hi, xl)),
        );
        acc[2] = veorq_u64(acc[2], pmull_u64(eq.hi, xh));
    }
}

/// Fold an [`UnredAcc`] to `(r0..r3)` and reduce mod p (same math as
/// [`crate::field::F256Unreduced::reduce`], so results are bit-identical).
#[inline]
fn reduce_unred(acc: &UnredAcc) -> F128 {
    unsafe {
        F256Unreduced {
            r0: vgetq_lane_u64::<0>(acc[0]),
            r1: vgetq_lane_u64::<1>(acc[0]) ^ vgetq_lane_u64::<0>(acc[1]),
            r2: vgetq_lane_u64::<0>(acc[2]) ^ vgetq_lane_u64::<1>(acc[1]),
            r3: vgetq_lane_u64::<1>(acc[2]),
        }
        .reduce()
    }
}

/// PROTOTYPE fusion of [`product_bs`] + [`fold_bs`] with NEON-resident LAZY
/// REDUCTION: each product coordinate is formed in a register and eq-multiplied
/// unreduced (4 PMULL + 4 VEOR, no mod-p fold, no lane extracts) into a vector
/// accumulator; the caller reduces each coordinate once per chunk. Kills the
/// 2.5 KB `prod` buffer AND the per-block reduction work.
#[inline(always)]
unsafe fn product_fold_bs(
    af: &[uint8x16_t; 160],
    bf: &[uint8x16_t; 160],
    ax: &[uint8x16_t; 64],
    bx: &[uint8x16_t; 64],
    eq: F128,
    res: &mut [UnredAcc; 160],
) {
    unsafe {
        #[inline(always)]
        unsafe fn acc(res: &mut UnredAcc, eq: F128, pr: uint8x16_t) {
            unsafe {
                mul_acc_unred(res, eq, vreinterpretq_u64_u8(pr));
            }
        }
        for p in 0..64 {
            let pr = veorq_u8(vandq_u8(af[p], bx[p]), vandq_u8(ax[p], bf[p]));
            acc(&mut res[p], eq, pr);
        }
        for p in 0..64 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[64 + p], bx[p]), vandq_u8(af[p], bf[p])),
                vandq_u8(ax[p], bf[64 + p]),
            );
            acc(&mut res[64 + p], eq, pr);
        }
        for p in 0..32 {
            let pr = veorq_u8(
                veorq_u8(vandq_u8(af[128 + p], bx[p]), vandq_u8(af[64 + p], bf[p])),
                veorq_u8(vandq_u8(af[p], bf[64 + p]), vandq_u8(ax[p], bf[128 + p])),
            );
            acc(&mut res[128 + p], eq, pr);
        }
    }
}

/// PROTOTYPE fusion of the paired-c transpose + [`fold_c`]: identical to
/// [`transpose_128x128_2src`] on `(c[base0], c[base1])`, but each output plane
/// is eq-folded into `wbar` straight from its register in pass B — the 2 KB
/// `pc` plane buffer (write + re-read per pair) disappears. Planes `c*8+k` for
/// `c < 8` are block 0's (fold with `eq0`), `c >= 8` are block 1's (`eq1`).
fn transpose_fold_c_2src(
    c_packed: &[u8],
    base0: usize,
    base1: usize,
    eq0: F128,
    eq1: F128,
    wbar: &mut [UnredAcc; 64],
) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                let row = gi * 8 + k;
                let l = vld1_u8(c_packed.as_ptr().add(base0 + row * 8));
                let h = vld1_u8(c_packed.as_ptr().add(base1 + row * 8));
                q[k] = vcombine_u8(l, h);
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                let x = vreinterpretq_u64_u8(v[c]);
                let j = c * 8 + k;
                if c < 8 {
                    mul_acc_unred(&mut wbar[j], eq0, x);
                } else {
                    mul_acc_unred(&mut wbar[j - 64], eq1, x);
                }
            }
        }
    }
}

fn encode_c(wbar: &[F128; 64]) -> [F128; 160] {
    // once: fresh = M * wbar
    let mut fresh = [F128::ZERO; 160];
    for lam in 0..160 {
        let mut m = M_MASK[lam];
        let mut acc = F128::ZERO;
        while m != 0 {
            acc += wbar[m.trailing_zeros() as usize];
            m &= m - 1;
        }
        fresh[lam] = acc;
    }
    fresh
}

fn deferred_c(cp: &[[uint8x16_t; 64]], eq: &[F128], n: usize) -> [F128; 160] {
    let mut wbar = [F128::ZERO; 64];
    for o in 0..n {
        unsafe {
            fold_c(&cp[o], eq[o], &mut wbar);
        }
    }
    encode_c(&wbar)
}

pub fn bin_abc_bitslice(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 160]) {
    let z = unsafe { vdupq_n_u8(0) };
    let mut res = [F128::ZERO; 160]; // AB (product) fold
    let mut wbar = [F128::ZERO; 64]; // C (linear) fold
    let mut af = [z; 160];
    let mut bf = [z; 160];
    let mut prod = [z; 160];
    for o in 0..n {
        encode_slp(&a_pl[o], &mut af);
        encode_slp(&b_pl[o], &mut bf);
        unsafe {
            product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
            fold_bs(&prod, eq[o], &mut res);
            fold_c(&c_pl[o], eq[o], &mut wbar);
        }
    }
    (res, encode_c(&wbar))
}

/// Multi-threaded fused AB+C: chunk the blocks, run the byte-identical serial
/// kernel per chunk (scratch allocated once per chunk), then reduce the
/// per-chunk (AB, C-`wbar`) accumulators by lane-wise F128 add. The fold is
/// associative + commutative, so the result is bit-identical to serial.
pub fn bin_abc_bitslice_par(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 160]) {
    use rayon::prelude::*;
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(8 * nthreads).max(1);
    let nchunks = n.div_ceil(chunk);
    let (res, wbar) = (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut wbar = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            for o in start..end {
                encode_slp(&a_pl[o], &mut af);
                encode_slp(&b_pl[o], &mut bf);
                unsafe {
                    product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
                    fold_bs(&prod, eq[o], &mut res);
                    fold_c(&c_pl[o], eq[o], &mut wbar);
                }
            }
            (res, wbar)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64]),
            |(mut r1, mut w1), (r2, w2)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    w1[k] += w2[k];
                }
                (r1, w1)
            },
        );
    (res, encode_c(&wbar))
}

// ---------------------------------------------------------------------------
// Production path: encode against `M` derived from the M2 evaluator's own
// extension, so the kernel emits coordinates in the evaluator's basis (identity
// bridge to `product_code_message`). The bench `M_MASK`/`encode_slp` above are
// the legacy SLP for a *different coordinate labeling* of the same code and are
// kept only for the SLP-parity test; do not use them in production.
// ---------------------------------------------------------------------------

/// The by-point fresh encode `M` (160×64) for the genus-95 product code, derived
/// once from the M2 evaluator's extension. Fresh slot `s` (0..158) == evaluator
/// product coord `64 + s` (order1|order2|order3); slots 158,159 are D³ garbage
/// (only 30 D³ points exist). Built lazily from `extended_base_product_message`.
fn derived_m() -> &'static [u64; 160] {
    static M: OnceLock<[u64; 160]> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = [0u64; 160];
        for j in 0..64 {
            let ext = extended_base_product_message(BaseMessage(1u64 << j));
            for s in 0..158 {
                if ext.get_bit(64 + s) {
                    m[s] |= 1u64 << j;
                }
            }
        }
        m
    })
}

/// Direct bitsliced encode `out = M · inp`: one XOR per set bit of each row of
/// `M`. Code-instance-agnostic (works for the derived `M`); replaces the bench
/// Paar SLP, which is hardwired to the legacy `M_MASK`. Regenerating an optimized
/// SLP for `derived_m` is a deferred perf step.
#[inline]
unsafe fn encode_direct(m: &[u64; 160], inp: &[uint8x16_t; 64], out: &mut [uint8x16_t; 160]) {
    unsafe {
        for s in 0..160 {
            let mut acc = vdupq_n_u8(0);
            let mut mask = m[s];
            while mask != 0 {
                let j = mask.trailing_zeros() as usize;
                acc = veorq_u8(acc, inp[j]);
                mask &= mask - 1;
            }
            out[s] = acc;
        }
    }
}

/// `fresh = M · wbar` for the linear C path, against the derived `M`.
fn encode_c_derived(wbar: &[F128; 64]) -> [F128; 160] {
    let m = derived_m();
    let mut fresh = [F128::ZERO; 160];
    for lam in 0..160 {
        let mut mm = m[lam];
        let mut acc = F128::ZERO;
        while mm != 0 {
            acc += wbar[mm.trailing_zeros() as usize];
            mm &= mm - 1;
        }
        fresh[lam] = acc;
    }
    fresh
}

/// Production fused AB+C round-1 message over `n` pre-bitsliced blocks, encoding
/// against the evaluator-derived `M`. Returns the 160 by-point FRESH coords for
/// AB and for C in the evaluator's basis: fresh slot `s` (0..158) is evaluator
/// product coord `64 + s`; slots 158,159 are ignored garbage.
pub fn bin_abc(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 160]) {
    let m = derived_m();
    let z = unsafe { vdupq_n_u8(0) };
    let mut res = [F128::ZERO; 160];
    let mut wbar = [F128::ZERO; 64];
    let mut af = [z; 160];
    let mut bf = [z; 160];
    let mut prod = [z; 160];
    for o in 0..n {
        unsafe {
            encode_direct(m, &a_pl[o], &mut af);
            encode_direct(m, &b_pl[o], &mut bf);
            product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
            fold_bs(&prod, eq[o], &mut res);
            fold_c(&c_pl[o], eq[o], &mut wbar);
        }
    }
    (res, encode_c_derived(&wbar))
}

/// Multi-threaded [`bin_abc`]: split the blocks into contiguous chunks, run the
/// serial derived-`M` kernel on each (scratch allocated once per chunk), then
/// reduce the per-chunk `(AB, C-wbar)` accumulators by lane-wise F128 add. The
/// fold is associative + commutative, so the result is bit-identical to serial.
pub fn bin_abc_par(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 160]) {
    use rayon::prelude::*;
    let m = derived_m();
    let nthreads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(8 * nthreads).max(1);
    let nchunks = n.div_ceil(chunk);
    let (res, wbar) = (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut wbar = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            for o in start..end {
                unsafe {
                    encode_direct(m, &a_pl[o], &mut af);
                    encode_direct(m, &b_pl[o], &mut bf);
                    product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
                    fold_bs(&prod, eq[o], &mut res);
                    fold_c(&c_pl[o], eq[o], &mut wbar);
                }
            }
            (res, wbar)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64]),
            |(mut r1, mut w1), (r2, w2)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    w1[k] += w2[k];
                }
                (r1, w1)
            },
        );
    (res, encode_c_derived(&wbar))
}

/// Read a bit-packed witness (LSB-first) into pre-bitsliced 128-message blocks.
///
/// AG variable order (derived from the RS packed layout in
/// `univariate_skip_optimized`): **skip = low 6 bits**, **inner = next 7 bits**,
/// **outer = high `m−13` bits**. So one 64-bit skip-message is 8 contiguous LE
/// bytes and a 128-message block is 1024 contiguous bytes. Message `i` within a
/// block is the inner position weighted `γ^i` by the kernel's within-block
/// reinterpret (the geometric-progression friendly challenges). Whether this
/// ordering matches the commitment is confirmed end-to-end in M3.
pub fn blocks_from_packed(packed: &[u8]) -> Vec<[uint8x16_t; 64]> {
    assert_eq!(
        packed.len() % 1024,
        0,
        "packed witness must be a whole number of 128-message (1024-byte) blocks"
    );
    let n = packed.len() / 1024;
    let z = unsafe { vdupq_n_u8(0) };
    // NEON bit-transpose, pairing two 64-bit-message blocks into one 128-wide
    // `transpose_128x128` so the transpose runs at full utilization (no zero
    // padding). Halves the transpose calls vs one-block-per-pass.
    let mut out: Vec<[uint8x16_t; 64]> = Vec::with_capacity(n);
    let mut buf = [0u8; 128 * 16];
    let mut planes = [z; 128];
    let mut o = 0;
    while o + 1 < n {
        let (b0, b1) = (o * 1024, (o + 1) * 1024);
        for r in 0..128 {
            buf[r * 16..r * 16 + 8].copy_from_slice(&packed[b0 + r * 8..b0 + r * 8 + 8]);
            buf[r * 16 + 8..r * 16 + 16].copy_from_slice(&packed[b1 + r * 8..b1 + r * 8 + 8]);
        }
        transpose_128x128(&buf, &mut planes);
        let mut p0 = [z; 64];
        p0.copy_from_slice(&planes[0..64]);
        out.push(p0);
        let mut p1 = [z; 64];
        p1.copy_from_slice(&planes[64..128]);
        out.push(p1);
        o += 2;
    }
    if o < n {
        // Final odd block: pad the high half with zeros.
        let b0 = o * 1024;
        for r in 0..128 {
            buf[r * 16..r * 16 + 8].copy_from_slice(&packed[b0 + r * 8..b0 + r * 8 + 8]);
            buf[r * 16 + 8..r * 16 + 16].fill(0);
        }
        transpose_128x128(&buf, &mut planes);
        let mut p0 = [z; 64];
        p0.copy_from_slice(&planes[0..64]);
        out.push(p0);
    }
    out
}

/// Convenience: read the three packed witnesses and run the parallel fused
/// AB+C kernel. `eq` has one outer weight per 1024-byte block. The `C`-scaling
/// and `r`-derived `eq` live in the M3 protocol layer.
pub fn bin_abc_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 160]) {
    let ap = blocks_from_packed(a_packed);
    let bp = blocks_from_packed(b_packed);
    let cp = blocks_from_packed(c_packed);
    let n = ap.len();
    assert_eq!(eq.len(), n, "one eq weight per block");
    bin_abc_par(&ap, &bp, &cp, eq, n)
}

/// Raw fused AB+C round-1 over pre-bitsliced blocks: returns the AB product
/// fresh coords (160, D-scaled; slots 158/159 garbage) and the folded C message
/// `wbar` (64, D-scaled) WITHOUT encoding C. The AG-skip protocol sends `wbar`
/// (the c message) directly, not its codeword, so the prover wants it raw.
pub fn round1_raw(
    a_pl: &[[uint8x16_t; 64]],
    b_pl: &[[uint8x16_t; 64]],
    c_pl: &[[uint8x16_t; 64]],
    eq: &[F128],
    n: usize,
) -> ([F128; 160], [F128; 64]) {
    let m = derived_m();
    let z = unsafe { vdupq_n_u8(0) };
    let mut res = [F128::ZERO; 160];
    let mut wbar = [F128::ZERO; 64];
    let mut af = [z; 160];
    let mut bf = [z; 160];
    let mut prod = [z; 160];
    for o in 0..n {
        unsafe {
            encode_direct(m, &a_pl[o], &mut af);
            encode_direct(m, &b_pl[o], &mut bf);
            product_bs(&af, &bf, &a_pl[o], &b_pl[o], &mut prod);
            fold_bs(&prod, eq[o], &mut res);
            fold_c(&c_pl[o], eq[o], &mut wbar);
        }
    }
    (res, wbar)
}

/// [`round1_raw`] reading the three packed witnesses (one `eq` per 1024-byte block).
pub fn round1_raw_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    crate::suboptimal_path!(
        "reference round-1 (raw, non-bitsliced)",
        "round1_slp_packed_banks_fused"
    );
    let ap = blocks_from_packed(a_packed);
    let bp = blocks_from_packed(b_packed);
    let cp = blocks_from_packed(c_packed);
    let n = ap.len();
    assert_eq!(eq.len(), n, "one eq weight per block");
    round1_raw(&ap, &bp, &cp, eq, n)
}

// ---------------------------------------------------------------------------
// Fast derived-`M` encode: four-Russians LUT (row-major) + in-register product.
// This replaces `encode_direct` on the production path (~10× faster). The LUT is
// built from `derived_m`, so the kernel stays in the evaluator's basis (identity
// bridge). Same four-Russians blocking the M2 evaluator already uses.
// ---------------------------------------------------------------------------

/// `8×256×20`-byte four-Russians table for the derived `M`: entry `(pos, byte)`
/// is the 160-bit (20-byte) encode contribution of input byte `byte` at byte
/// position `pos`. Built once.
fn derived_lut() -> &'static [u8] {
    static LUT: OnceLock<Vec<u8>> = OnceLock::new();
    LUT.get_or_init(|| {
        let m = derived_m();
        let mut t = vec![0u8; 8 * 256 * 20];
        for pos in 0..8 {
            for byte in 0..256usize {
                let e = (pos * 256 + byte) * 20;
                for k in 0..160 {
                    if (((m[k] >> (pos * 8)) & byte as u64).count_ones() & 1) == 1 {
                        t[e + k / 8] |= 1u8 << (k % 8);
                    }
                }
            }
        }
        t
    })
}

/// Encode one 64-bit message via the four-Russians table: 8 byte-lookups XORed,
/// keeping the 128-bit D1+D2 body in a register and the 32-bit D3 tail separate.
#[inline(always)]
unsafe fn encode_lut_v(table: *const u8, row: &[u8; 8]) -> (uint8x16_t, u32) {
    unsafe {
        let mut acc = vdupq_n_u8(0);
        let mut tail = 0u32;
        for pos in 0..8 {
            let base = (pos * 256 + row[pos] as usize) * 20;
            acc = veorq_u8(acc, vld1q_u8(table.add(base)));
            tail ^= (table.add(base + 16) as *const u32).read_unaligned();
        }
        (acc, tail)
    }
}

/// Transpose 128 rows × 160 bits (each row 20 bytes: 16 body + 4 tail) into 160
/// planes × 128 bits (each plane 16 bytes). Generic (no `M` dependency).
fn transpose_128x160_hybrid(src: &[u8], dst: &mut [u8]) {
    const SS: usize = 20;
    const DS: usize = 16;
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let (m4d, m2d, m1d) = (vdup_n_u8(0x0F), vdup_n_u8(0x33), vdup_n_u8(0x55));
        let mut ws = [0u8; 8 * 16 * 16];
        for big_i in 0..16usize {
            let so = 8 * big_i * SS;
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                q[k] = vld1q_u8(src.as_ptr().add(so + k * SS));
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + big_i) * 16), q[k]);
            }
            let mut d = [vdup_n_u8(0); 8];
            for k in 0..8 {
                let mut tl = [0u8; 4];
                tl.copy_from_slice(&src[so + k * SS + 16..so + k * SS + 20]);
                d[k] =
                    vreinterpret_u8_u32(vset_lane_u32::<0>(u32::from_le_bytes(tl), vdup_n_u32(0)));
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = d[a];
                d[a] = vbsl_u8(m4d, t, vshl_n_u8::<4>(d[b]));
                d[b] = vbsl_u8(m4d, vshr_n_u8::<4>(t), d[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = d[a];
                d[a] = vbsl_u8(m2d, t, vshl_n_u8::<2>(d[b]));
                d[b] = vbsl_u8(m2d, vshr_n_u8::<2>(t), d[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = d[a];
                d[a] = vbsl_u8(m1d, t, vshl_n_u8::<1>(d[b]));
                d[b] = vbsl_u8(m1d, vshr_n_u8::<1>(t), d[b]);
            }
            let mut t8 = [[0u8; 8]; 8];
            for k in 0..8 {
                vst1_u8(t8[k].as_mut_ptr(), d[k]);
            }
            for j in 0..4 {
                for k in 0..8 {
                    dst[(8 * (16 + j) + k) * DS + big_i] = t8[k][j];
                }
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                vst1q_u8(dst.as_mut_ptr().add((c * 8 + k) * DS), v[c]);
            }
        }
    }
}

/// Production fused AB+C round-1 via the four-Russians LUT encode (the fast
/// path). Reads `a`/`b` row-major (LUT-encode → in-register product → transpose
/// → fold) and `c` bit-sliced (linear `fold_c` + one deferred `encode_c`).
/// Output matches [`round1_raw_packed`] on the 158 real fresh coords.
pub fn round1_lut_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    let n = a_packed.len() / 1024;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let table = derived_lut().as_ptr();
    let cp = blocks_from_packed(c_packed);
    let mut res = [F128::ZERO; 160];
    let mut wbar = [F128::ZERO; 64];
    let mut rm = vec![0u8; 128 * 20];
    let mut block = [F128::ZERO; 160];
    for o in 0..n {
        let base = o * 1024;
        for r in 0..128 {
            let ma =
                u64::from_le_bytes(a_packed[base + r * 8..base + r * 8 + 8].try_into().unwrap());
            let mb =
                u64::from_le_bytes(b_packed[base + r * 8..base + r * 8 + 8].try_into().unwrap());
            unsafe {
                let (aa, at) = encode_lut_v(table, &ma.to_le_bytes());
                let (ba, bt) = encode_lut_v(table, &mb.to_le_bytes());
                let x1 = vreinterpretq_u8_u64(vdupq_n_u64(ma));
                let x2 = vreinterpretq_u8_u64(vdupq_n_u64(mb));
                // D1 (low 64) + D2 (high 64); the a1&b1 cross-term lifted into D2.
                let cross = vcombine_u8(vdup_n_u8(0), vget_low_u8(vandq_u8(aa, ba)));
                let pr_acc = veorq_u8(veorq_u8(vandq_u8(aa, x2), vandq_u8(x1, ba)), cross);
                // D3 (32-bit tail) on the relevant lanes.
                let au = vreinterpretq_u32_u8(aa);
                let bu = vreinterpretq_u32_u8(ba);
                let a1 = vgetq_lane_u32::<0>(au);
                let a2d = vgetq_lane_u32::<2>(au);
                let b1 = vgetq_lane_u32::<0>(bu);
                let b2d = vgetq_lane_u32::<2>(bu);
                let pr_tail = (at & (mb as u32)) ^ (a2d & b1) ^ (a1 & b2d) ^ ((ma as u32) & bt);
                vst1q_u8(rm.as_mut_ptr().add(r * 20), pr_acc);
                (rm.as_mut_ptr().add(r * 20 + 16) as *mut u32).write_unaligned(pr_tail);
            }
        }
        let bview =
            unsafe { std::slice::from_raw_parts_mut(block.as_mut_ptr() as *mut u8, 160 * 16) };
        transpose_128x160_hybrid(&rm, bview);
        for j in 0..160 {
            res[j] += eq[o] * block[j];
        }
        unsafe {
            fold_c(&cp[o], eq[o], &mut wbar);
        }
    }
    (res, wbar)
}

/// Round-1 via the Paar straight-line encode ([`super::slp_derived`]): bit-slice
/// a/b/c, run the SLP on planes, product, fold. No *output* transpose (the SLP
/// works on planes directly; it pays an *input* transpose instead). Output
/// matches [`round1_raw_packed`] on the 158 fresh coords + `wbar`.
pub fn round1_slp_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    crate::suboptimal_path!("unfused SLP round-1", "round1_slp_packed_banks_fused");
    use rayon::prelude::*;
    let n = a_packed.len() / 1024;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = rayon::current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1); // ~8 chunks/thread
    let chunk = chunk0 + (chunk0 & 1); // even, so the block-pair loop tiles each chunk exactly
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut wbar = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            // Per-chunk bitslice scratch. Process blocks in PAIRS so every
            // transpose_128x128 is fully used (128 input columns, no zero pad):
            // a+b of a block pair into one transpose (a→[0..64], b→[64..128]); the
            // two blocks' c pair into one more (c_o→[0..64], c_{o+1}→[64..128]).
            // 3 transposes per 2 blocks vs 4 with c padded per-block.
            let mut pab = [z; 128];
            let mut pc = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * 1024, (o + 1) * 1024);
                // The two blocks' c paired straight from packed (no interleave buf):
                // pc[0..64] = block o's c, pc[64..128] = block o+1's c.
                transpose_128x128_2src(c_packed, cb0, c_packed, cb1, &mut pc);
                let cp0: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                let cp1: &[uint8x16_t; 64] = pc[64..128].try_into().unwrap();
                unsafe {
                    process_block(
                        a_packed, b_packed, cb0, eq[o], cp0, &mut pab, &mut af, &mut bf, &mut prod,
                        &mut res, &mut wbar,
                    );
                    process_block(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        cp1,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut wbar,
                    );
                }
                o += 2;
            }
            if o < end {
                // Trailing odd block (only n odd, i.e. m=13): c padded via buf.
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * 1024, &mut buf, &mut pc);
                let cp: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                unsafe {
                    process_block(
                        a_packed,
                        b_packed,
                        o * 1024,
                        eq[o],
                        cp,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut wbar,
                    );
                }
            }
            (res, wbar)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64]),
            |(mut r1, mut w1), (r2, w2)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    w1[k] += w2[k];
                }
                (r1, w1)
            },
        )
}

/// One block of the SLP round-1: pair a+b into one transpose (a→`pab[0..64]`,
/// b→`pab[64..128]`), then encode·product·fold into `res` and fold c (`cp`, the
/// caller-supplied c planes) into `wbar`. The accumulators are shared across the
/// chunk's blocks; `buf`/`pab` are reused scratch.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_block(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    cp: &[uint8x16_t; 64],
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    prod: &mut [uint8x16_t; 160],
    res: &mut [F128; 160],
    wbar: &mut [F128; 64],
) {
    // a+b straight from the packed witnesses into one transpose (no interleave buf).
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        super::slp_derived::encode_slp_derived(ap, af);
        super::slp_derived::encode_slp_derived(bp, bf);
        product_bs(af, bf, ap, bp, prod);
        fold_bs(prod, eq_o, res);
        fold_c(cp, eq_o, wbar);
    }
}

/// PROTOTYPE [`process_block`] with the fused product+fold ([`product_fold_bs`])
/// — no `prod` buffer. The c-path is handled separately by the caller via
/// [`transpose_fold_c_2src`], so `cp` is gone too.
#[inline]
unsafe fn process_block_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    res: &mut [UnredAcc; 160],
) {
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        super::slp_derived::encode_slp_derived(ap, af);
        super::slp_derived::encode_slp_derived(bp, bf);
        product_fold_bs(af, bf, ap, bp, eq_o, res);
    }
}

/// PROTOTYPE [`round1_slp_packed`] with buffer-pass fusion: the `prod` buffer is
/// gone ([`product_fold_bs`]) and the paired c-planes are eq-folded straight out
/// of the transpose's pass-B registers ([`transpose_fold_c_2src`]), so the `pc`
/// buffer is gone. Output is bit-identical to [`round1_slp_packed`].
pub fn round1_slp_packed_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64]) {
    crate::suboptimal_path!(
        "prototype fused round-1 (no banks)",
        "round1_slp_packed_banks_fused"
    );
    use rayon::prelude::*;
    let n = a_packed.len() / 1024;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = rayon::current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let z64 = unsafe { vdupq_n_u64(0) };
            let mut res = [[z64; 3]; 160];
            let mut wbar = [[z64; 3]; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut pab = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * 1024, (o + 1) * 1024);
                transpose_fold_c_2src(c_packed, cb0, cb1, eq[o], eq[o + 1], &mut wbar);
                unsafe {
                    process_block_fused(
                        a_packed, b_packed, cb0, eq[o], &mut pab, &mut af, &mut bf, &mut res,
                    );
                    process_block_fused(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                }
                o += 2;
            }
            if o < end {
                // Trailing odd block: c padded via buf, unreduced fold_c inline.
                let mut pc = [z; 128];
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * 1024, &mut buf, &mut pc);
                unsafe {
                    process_block_fused(
                        a_packed,
                        b_packed,
                        o * 1024,
                        eq[o],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                    for k in 0..64 {
                        mul_acc_unred(&mut wbar[k], eq[o], vreinterpretq_u64_u8(pc[k]));
                    }
                }
            }
            // Reduce once per chunk (amortized over the chunk's blocks).
            let mut res_r = [F128::ZERO; 160];
            let mut wbar_r = [F128::ZERO; 64];
            for j in 0..160 {
                res_r[j] = reduce_unred(&res[j]);
            }
            for k in 0..64 {
                wbar_r[k] = reduce_unred(&wbar[k]);
            }
            (res_r, wbar_r)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64]),
            |(mut r1, mut w1), (r2, w2)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    w1[k] += w2[k];
                }
                (r1, w1)
            },
        )
}

/// Two-bank variant of [`fold_c`] for `s_hat_v_c` capture: split each plane's
/// `pf = Σ_i x^i·bit_i` by the parity of the friendly index `i` (= the 7th
/// packing bit / friendly bit 0) into `bank0` (even `i`) and `bank1` (odd `i`).
/// Since the even/odd bit sets partition `pf`, `bank0[k] + bank1[k]` equals
/// [`fold_c`]'s `wbar[k]` bit-for-bit (XOR partition + field-mult distributivity).
const C_EVEN_MASK: u64 = 0x5555_5555_5555_5555; // bits 0,2,4,…
const C_ODD_MASK: u64 = 0xAAAA_AAAA_AAAA_AAAA; // bits 1,3,5,…
unsafe fn fold_c_banks(
    cp: &[uint8x16_t; 64],
    eq: F128,
    bank0: &mut [F128; 64],
    bank1: &mut [F128; 64],
) {
    unsafe {
        for k in 0..64 {
            let pf = vreinterpretq_u64_u8(cp[k]);
            let lo = vgetq_lane_u64::<0>(pf);
            let hi = vgetq_lane_u64::<1>(pf);
            let even = F128 {
                lo: lo & C_EVEN_MASK,
                hi: hi & C_EVEN_MASK,
            };
            let odd = F128 {
                lo: lo & C_ODD_MASK,
                hi: hi & C_ODD_MASK,
            };
            bank0[k] += eq * even;
            bank1[k] += eq * odd;
        }
    }
}

/// [`process_block`] with the two-bank c-fold ([`fold_c_banks`]) for `s_hat_v_c`
/// capture. The AB path is identical; only the c accumulation differs.
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn process_block_banks(
    a_packed: &[u8],
    b_packed: &[u8],
    base: usize,
    eq_o: F128,
    cp: &[uint8x16_t; 64],
    pab: &mut [uint8x16_t; 128],
    af: &mut [uint8x16_t; 160],
    bf: &mut [uint8x16_t; 160],
    prod: &mut [uint8x16_t; 160],
    res: &mut [F128; 160],
    bank0: &mut [F128; 64],
    bank1: &mut [F128; 64],
) {
    transpose_128x128_2src(a_packed, base, b_packed, base, pab);
    let ap: &[uint8x16_t; 64] = (&pab[0..64]).try_into().unwrap();
    let bp: &[uint8x16_t; 64] = (&pab[64..128]).try_into().unwrap();
    unsafe {
        super::slp_derived::encode_slp_derived(ap, af);
        super::slp_derived::encode_slp_derived(bp, bf);
        product_bs(af, bf, ap, bp, prod);
        fold_bs(prod, eq_o, res);
        fold_c_banks(cp, eq_o, bank0, bank1);
    }
}

/// [`round1_slp_packed`] that ALSO returns the two c-fold banks for `s_hat_v_c`
/// capture (split by the 7th packing bit). `res` and `bank0 + bank1` are
/// bit-identical to `round1_slp_packed`'s `(res, wbar)`. Kept separate from the
/// hot `round1_slp_packed` so the standalone round-1 microbench path is
/// untouched; production AG prove (which needs `s_hat_v_c`) calls this.
pub fn round1_slp_packed_banks(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64], [F128; 64]) {
    crate::suboptimal_path!("unfused banks round-1", "round1_slp_packed_banks_fused");
    use rayon::prelude::*;
    let n = a_packed.len() / 1024;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = rayon::current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let mut res = [F128::ZERO; 160];
            let mut bank0 = [F128::ZERO; 64];
            let mut bank1 = [F128::ZERO; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut prod = [z; 160];
            let mut pab = [z; 128];
            let mut pc = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * 1024, (o + 1) * 1024);
                transpose_128x128_2src(c_packed, cb0, c_packed, cb1, &mut pc);
                let cp0: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                let cp1: &[uint8x16_t; 64] = pc[64..128].try_into().unwrap();
                unsafe {
                    process_block_banks(
                        a_packed, b_packed, cb0, eq[o], cp0, &mut pab, &mut af, &mut bf, &mut prod,
                        &mut res, &mut bank0, &mut bank1,
                    );
                    process_block_banks(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        cp1,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut bank0,
                        &mut bank1,
                    );
                }
                o += 2;
            }
            if o < end {
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * 1024, &mut buf, &mut pc);
                let cp: &[uint8x16_t; 64] = pc[0..64].try_into().unwrap();
                unsafe {
                    process_block_banks(
                        a_packed,
                        b_packed,
                        o * 1024,
                        eq[o],
                        cp,
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut prod,
                        &mut res,
                        &mut bank0,
                        &mut bank1,
                    );
                }
            }
            (res, bank0, bank1)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64], [F128::ZERO; 64]),
            |(mut r1, mut a0, mut a1), (r2, b0, b1)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    a0[k] += b0[k];
                    a1[k] += b1[k];
                }
                (r1, a0, a1)
            },
        )
}

/// [`transpose_fold_c_2src`] with the two-bank c-split of [`fold_c_banks`]:
/// each plane is masked into its even/odd-index halves in NEON registers and
/// each half is eq-multiplied unreduced into its bank. Same pass structure —
/// the `pc` buffer never exists.
fn transpose_fold_c_banks_2src(
    c_packed: &[u8],
    base0: usize,
    base1: usize,
    eq0: F128,
    eq1: F128,
    bank0: &mut [UnredAcc; 64],
    bank1: &mut [UnredAcc; 64],
) {
    unsafe {
        let (m4, m2, m1) = (vdupq_n_u8(0x0F), vdupq_n_u8(0x33), vdupq_n_u8(0x55));
        let even = vdupq_n_u64(C_EVEN_MASK);
        let odd = vdupq_n_u64(C_ODD_MASK);
        let mut ws = [0u8; 8 * 16 * 16];
        for gi in 0..16usize {
            let mut q = [vdupq_n_u8(0); 8];
            for k in 0..8 {
                let row = gi * 8 + k;
                let l = vld1_u8(c_packed.as_ptr().add(base0 + row * 8));
                let h = vld1_u8(c_packed.as_ptr().add(base1 + row * 8));
                q[k] = vcombine_u8(l, h);
            }
            for &(a, b) in &[(0, 4), (1, 5), (2, 6), (3, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m4, t, vshlq_n_u8::<4>(q[b]));
                q[b] = vbslq_u8(m4, vshrq_n_u8::<4>(t), q[b]);
            }
            for &(a, b) in &[(0, 2), (1, 3), (4, 6), (5, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m2, t, vshlq_n_u8::<2>(q[b]));
                q[b] = vbslq_u8(m2, vshrq_n_u8::<2>(t), q[b]);
            }
            for &(a, b) in &[(0, 1), (2, 3), (4, 5), (6, 7)] {
                let t = q[a];
                q[a] = vbslq_u8(m1, t, vshlq_n_u8::<1>(q[b]));
                q[b] = vbslq_u8(m1, vshrq_n_u8::<1>(t), q[b]);
            }
            for k in 0..8 {
                vst1q_u8(ws.as_mut_ptr().add((k * 16 + gi) * 16), q[k]);
            }
        }
        for k in 0..8usize {
            let mut v = [vdupq_n_u8(0); 16];
            for i in 0..16 {
                v[i] = vld1q_u8(ws.as_ptr().add((k * 16 + i) * 16));
            }
            transpose16x16(&mut v);
            for c in 0..16 {
                let x = vreinterpretq_u64_u8(v[c]);
                let j = c * 8 + k;
                let (jj, eq) = if c < 8 { (j, eq0) } else { (j - 64, eq1) };
                mul_acc_unred(&mut bank0[jj], eq, vandq_u64(x, even));
                mul_acc_unred(&mut bank1[jj], eq, vandq_u64(x, odd));
            }
        }
    }
}

/// PROTOTYPE fused [`round1_slp_packed_banks`]: fused product+fold (no `prod`
/// buffer), banked c-fold straight out of the c-transpose registers (no `pc`
/// buffer), NEON-resident lazy reduction (reduce once per chunk). Bit-identical
/// to [`round1_slp_packed_banks`] on `(res, bank0, bank1)`.
pub fn round1_slp_packed_banks_fused(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    eq: &[F128],
) -> ([F128; 160], [F128; 64], [F128; 64]) {
    use rayon::prelude::*;
    let n = a_packed.len() / 1024;
    assert_eq!(eq.len(), n, "one eq weight per block");
    let nthreads = rayon::current_num_threads().max(1);
    let chunk0 = n.div_ceil(8 * nthreads).max(1);
    let chunk = chunk0 + (chunk0 & 1);
    let nchunks = n.div_ceil(chunk);
    (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let start = ci * chunk;
            let end = ((ci + 1) * chunk).min(n);
            let z = unsafe { vdupq_n_u8(0) };
            let z64 = unsafe { vdupq_n_u64(0) };
            let mut res = [[z64; 3]; 160];
            let mut bank0 = [[z64; 3]; 64];
            let mut bank1 = [[z64; 3]; 64];
            let mut af = [z; 160];
            let mut bf = [z; 160];
            let mut pab = [z; 128];
            let mut o = start;
            while o + 1 < end {
                let (cb0, cb1) = (o * 1024, (o + 1) * 1024);
                transpose_fold_c_banks_2src(
                    c_packed,
                    cb0,
                    cb1,
                    eq[o],
                    eq[o + 1],
                    &mut bank0,
                    &mut bank1,
                );
                unsafe {
                    process_block_fused(
                        a_packed, b_packed, cb0, eq[o], &mut pab, &mut af, &mut bf, &mut res,
                    );
                    process_block_fused(
                        a_packed,
                        b_packed,
                        cb1,
                        eq[o + 1],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                }
                o += 2;
            }
            if o < end {
                let mut pc = [z; 128];
                let mut buf = [0u8; 128 * 16];
                bitslice_block_into(c_packed, o * 1024, &mut buf, &mut pc);
                unsafe {
                    process_block_fused(
                        a_packed,
                        b_packed,
                        o * 1024,
                        eq[o],
                        &mut pab,
                        &mut af,
                        &mut bf,
                        &mut res,
                    );
                    let even = vdupq_n_u64(C_EVEN_MASK);
                    let odd = vdupq_n_u64(C_ODD_MASK);
                    for k in 0..64 {
                        let x = vreinterpretq_u64_u8(pc[k]);
                        mul_acc_unred(&mut bank0[k], eq[o], vandq_u64(x, even));
                        mul_acc_unred(&mut bank1[k], eq[o], vandq_u64(x, odd));
                    }
                }
            }
            let mut res_r = [F128::ZERO; 160];
            let mut b0_r = [F128::ZERO; 64];
            let mut b1_r = [F128::ZERO; 64];
            for j in 0..160 {
                res_r[j] = reduce_unred(&res[j]);
            }
            for k in 0..64 {
                b0_r[k] = reduce_unred(&bank0[k]);
                b1_r[k] = reduce_unred(&bank1[k]);
            }
            (res_r, b0_r, b1_r)
        })
        .reduce(
            || ([F128::ZERO; 160], [F128::ZERO; 64], [F128::ZERO; 64]),
            |(mut r1, mut a0, mut a1), (r2, b0, b1)| {
                for j in 0..160 {
                    r1[j] += r2[j];
                }
                for k in 0..64 {
                    a0[k] += b0[k];
                    a1[k] += b1[k];
                }
                (r1, a0, a1)
            },
        )
}

/// Bit-slice one 1024-byte block at `base` into 64 low planes (the high 64 of the
/// 128-wide transpose are the zero pad). `buf`'s high 8 bytes/row stay zero.
#[inline]
fn bitslice_block_into(
    packed: &[u8],
    base: usize,
    buf: &mut [u8; 128 * 16],
    planes: &mut [uint8x16_t; 128],
) {
    for r in 0..128 {
        buf[r * 16..r * 16 + 8].copy_from_slice(&packed[base + r * 8..base + r * 8 + 8]);
    }
    transpose_128x128(buf, planes);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn n(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    fn par(mask: u64, msg: u64) -> bool {
        (mask & msg).count_ones() & 1 == 1
    }

    fn bitslice(msgs: &[u64; 128]) -> [uint8x16_t; 64] {
        let mut planes = [[0u8; 16]; 64];
        for r in 0..128 {
            for k in 0..64 {
                if (msgs[r] >> k) & 1 == 1 {
                    planes[k][r / 8] |= 1u8 << (r % 8);
                }
            }
        }
        let mut out = [unsafe { vdupq_n_u8(0) }; 64];
        for k in 0..64 {
            out[k] = unsafe { vld1q_u8(planes[k].as_ptr()) };
        }
        out
    }

    fn scalar_ref(
        a_msg: &[[u64; 128]],
        b_msg: &[[u64; 128]],
        eq: &[F128],
        n: usize,
    ) -> [F128; 160] {
        let mut res = [F128::ZERO; 160];
        for o in 0..n {
            let mut word = [F128::ZERO; 160];
            for r in 0..128 {
                let am = a_msg[o][r];
                let bm = b_msg[o][r];
                let mut af = [false; 160];
                let mut bf = [false; 160];
                for k in 0..160 {
                    af[k] = par(M_MASK[k], am);
                    bf[k] = par(M_MASK[k], bm);
                }
                let mut pr = [false; 160];
                for p in 0..64 {
                    pr[p] = (af[p] & ((bm >> p) & 1 == 1)) ^ (((am >> p) & 1 == 1) & bf[p]);
                }
                for p in 0..64 {
                    pr[64 + p] = (af[64 + p] & ((bm >> p) & 1 == 1))
                        ^ (af[p] & bf[p])
                        ^ (((am >> p) & 1 == 1) & bf[64 + p]);
                }
                for p in 0..32 {
                    pr[128 + p] = (af[128 + p] & ((bm >> p) & 1 == 1))
                        ^ (af[64 + p] & bf[p])
                        ^ (af[p] & bf[64 + p])
                        ^ (((am >> p) & 1 == 1) & bf[128 + p]);
                }
                for j in 0..160 {
                    if pr[j] {
                        if r < 64 {
                            word[j].lo |= 1u64 << r;
                        } else {
                            word[j].hi |= 1u64 << (r - 64);
                        }
                    }
                }
            }
            for j in 0..160 {
                res[j] += eq[o] * word[j];
            }
        }
        res
    }

    fn scalar_c(cm: &[[u64; 128]], eq: &[F128], n: usize) -> [F128; 160] {
        let mut res = [F128::ZERO; 160];
        for o in 0..n {
            let mut word = [F128::ZERO; 160];
            for r in 0..128 {
                let c = cm[o][r];
                for lam in 0..160 {
                    if par(M_MASK[lam], c) {
                        if r < 64 {
                            word[lam].lo |= 1u64 << r;
                        } else {
                            word[lam].hi |= 1u64 << (r - 64);
                        }
                    }
                }
            }
            for lam in 0..160 {
                res[lam] += eq[o] * word[lam];
            }
        }
        res
    }

    #[test]
    fn fused_abc_matches_scalar_and_par() {
        let mut rng = Rng(0xC0DE);
        let vn = 4usize;
        let mut am = vec![[0u64; 128]; vn];
        let mut bm = vec![[0u64; 128]; vn];
        let mut cm = vec![[0u64; 128]; vn];
        for o in 0..vn {
            for r in 0..128 {
                am[o][r] = rng.n();
                bm[o][r] = rng.n();
                cm[o][r] = rng.n();
            }
        }
        let ap: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&am[o])).collect();
        let bp: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&bm[o])).collect();
        let cp: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&cm[o])).collect();
        let eq: Vec<F128> = (0..vn)
            .map(|_| F128 {
                lo: rng.n(),
                hi: rng.n(),
            })
            .collect();

        let (fab, fc) = bin_abc_bitslice(&ap, &bp, &cp, &eq, vn);
        let want_ab = scalar_ref(&am, &bm, &eq, vn);
        let want_c = scalar_c(&cm, &eq, vn);
        assert!((0..160).all(|j| fab[j] == want_ab[j]), "AB != scalar_ref");
        assert!((0..160).all(|j| fc[j] == want_c[j]), "C != scalar_c");

        let (pab, pc) = bin_abc_bitslice_par(&ap, &bp, &cp, &eq, vn);
        assert!(
            (0..160).all(|j| pab[j] == fab[j] && pc[j] == fc[j]),
            "par != serial"
        );
    }

    /// One row's 160 by-point fresh product coords via the scalar formula
    /// (= `scalar_ref`'s inner loop), used to probe the kernel↔M2 correspondence.
    fn kernel_row_fresh(a: u64, b: u64) -> [bool; 160] {
        let mut af = [false; 160];
        let mut bf = [false; 160];
        for k in 0..160 {
            af[k] = par(M_MASK[k], a);
            bf[k] = par(M_MASK[k], b);
        }
        let mut pr = [false; 160];
        for p in 0..64 {
            pr[p] = (af[p] & ((b >> p) & 1 == 1)) ^ (((a >> p) & 1 == 1) & bf[p]);
        }
        for p in 0..64 {
            pr[64 + p] = (af[64 + p] & ((b >> p) & 1 == 1))
                ^ (af[p] & bf[p])
                ^ (((a >> p) & 1 == 1) & bf[64 + p]);
        }
        for p in 0..32 {
            pr[128 + p] = (af[128 + p] & ((b >> p) & 1 == 1))
                ^ (af[64 + p] & bf[p])
                ^ (af[p] & bf[64 + p])
                ^ (((a >> p) & 1 == 1) & bf[128 + p]);
        }
        pr
    }

    /// DIAGNOSTIC: the bench `M_MASK` and the M2 evaluator come from the same
    /// `succinctlabs/AG_codes` construction but use a DIFFERENT coordinate
    /// labeling (info-set / point ordering), so the kernel's 160 product forms,
    /// taken in the *raw witness bits*, span outside the evaluator's 222 forms —
    /// `rank(union) > rank(evaluator)`. This is a labeling mismatch, NOT a
    /// different code: deriving `M` from the evaluator's own extension makes the
    /// bridge the identity (see `m_derived_from_evaluator_is_identity_bridge`).
    /// Kept as the record of why the production `M` must come from the evaluator.
    #[test]
    fn kernel_vs_m2_evaluator_span() {
        use crate::genus95_curve_code::{BaseMessage, product_code_message};

        // Combined form rows: bits 0..222 evaluator coords, 222..382 kernel coords.
        const W: usize = 6; // 384 bits.
        fn rank(rows: &[[u64; W]], cols: std::ops::Range<usize>) -> usize {
            let n = cols.len();
            let mut m: Vec<[u64; W]> = rows
                .iter()
                .map(|r| {
                    let mut o = [0u64; W];
                    for (nc, c) in cols.clone().enumerate() {
                        if (r[c >> 6] >> (c & 63)) & 1 == 1 {
                            o[nc >> 6] |= 1u64 << (nc & 63);
                        }
                    }
                    o
                })
                .collect();
            let mut piv = 0usize;
            for c in 0..n {
                if let Some(p) = (piv..m.len()).find(|&p| (m[p][c >> 6] >> (c & 63)) & 1 == 1) {
                    m.swap(piv, p);
                    for r in 0..m.len() {
                        if r != piv && (m[r][c >> 6] >> (c & 63)) & 1 == 1 {
                            for i in 0..W {
                                m[r][i] ^= m[piv][i];
                            }
                        }
                    }
                    piv += 1;
                }
            }
            piv
        }

        const S: usize = 600;
        let mut rng = Rng(0xB0BA_CAFE_0000_0001);
        let mut rows: Vec<[u64; W]> = Vec::with_capacity(S);
        for _ in 0..S {
            let a = rng.n();
            let b = rng.n();
            let kf = kernel_row_fresh(a, b);
            let pm = product_code_message(BaseMessage(a), BaseMessage(b));
            let mut row = [0u64; W];
            for c in 0..222 {
                if pm.get_bit(c) {
                    row[c >> 6] |= 1u64 << (c & 63);
                }
            }
            for j in 0..160 {
                if kf[j] {
                    let c = 222 + j;
                    row[c >> 6] |= 1u64 << (c & 63);
                }
            }
            rows.push(row);
        }

        let rank_e = rank(&rows, 0..222);
        let rank_k = rank(&rows, 222..382);
        let rank_union = rank(&rows, 0..382);
        eprintln!(
            "[span] rank(evaluator 222)={rank_e}  rank(kernel 160)={rank_k}  \
             rank(union)={rank_union}  kernel-outside-evaluator={}",
            rank_union - rank_e
        );
        // The bench M_MASK's raw-bit forms span outside the evaluator's because
        // of the coordinate-labeling mismatch. Once the production M is derived
        // from the evaluator, the encode lives in the evaluator basis and this
        // diagnostic is superseded by the identity-bridge cross-check.
        assert!(
            rank_union > rank_e,
            "M_MASK now shares the evaluator basis — drop this diagnostic for the \
             derived-M identity-bridge cross-check"
        );
    }

    /// Derive the by-point fresh encode `M` from the M2 evaluator's OWN extension
    /// (`extended_base_product_message`), so the kernel speaks the evaluator's
    /// coordinate convention by construction. Slot `s` (0..158) maps to evaluator
    /// product coord `64 + s` (order1|order2|order3); slots 158,159 are D³ garbage
    /// (only 30 D³ points exist). The per-row product through this `M` must then
    /// equal `product_code_message` *with the identity bridge* — proving the bench
    /// `M_MASK` and the evaluator are the same code in a different coordinate
    /// labeling, and that deriving `M` from the evaluator reconciles them.
    #[test]
    fn m_derived_from_evaluator_is_identity_bridge() {
        use crate::genus95_curve_code::product::extended_base_product_message;
        use crate::genus95_curve_code::{BaseMessage, product_code_message};

        let mut m_eval = [0u64; 160];
        for j in 0..64 {
            let ext = extended_base_product_message(BaseMessage(1u64 << j));
            for s in 0..158 {
                if ext.get_bit(64 + s) {
                    m_eval[s] |= 1u64 << j;
                }
            }
        }

        let mut rng = Rng(0xD00D_F00D_0000_0001);
        for _ in 0..4096 {
            let a = rng.n();
            let b = rng.n();
            let mut af = [false; 160];
            let mut bf = [false; 160];
            for s in 0..160 {
                af[s] = (m_eval[s] & a).count_ones() & 1 == 1;
                bf[s] = (m_eval[s] & b).count_ones() & 1 == 1;
            }
            let mut pr = [false; 160];
            for p in 0..64 {
                pr[p] = (af[p] & ((b >> p) & 1 == 1)) ^ (((a >> p) & 1 == 1) & bf[p]);
            }
            for p in 0..64 {
                pr[64 + p] = (af[64 + p] & ((b >> p) & 1 == 1))
                    ^ (af[p] & bf[p])
                    ^ (((a >> p) & 1 == 1) & bf[64 + p]);
            }
            for p in 0..32 {
                pr[128 + p] = (af[128 + p] & ((b >> p) & 1 == 1))
                    ^ (af[64 + p] & bf[p])
                    ^ (af[p] & bf[64 + p])
                    ^ (((a >> p) & 1 == 1) & bf[128 + p]);
            }
            let pm = product_code_message(BaseMessage(a), BaseMessage(b));
            for s in 0..158 {
                assert_eq!(pr[s], pm.get_bit(64 + s), "coord {s} (a={a:#x} b={b:#x})");
            }
        }
    }

    /// M1 keystone: the production eq-folded kernel (`bin_abc`, encoding against
    /// the evaluator-derived `M`) must equal a reference built from the validated
    /// `product_code_message` (AB) and `extended_base_product_message` (C), under
    /// the identity bridge (fresh slot `s` == evaluator coord `64 + s`). `gamma^r`
    /// is the within-block geometric weight; `eq[o]` weights the blocks.
    #[test]
    fn bin_abc_matches_product_code_message() {
        use crate::genus95_curve_code::product::extended_base_product_message;
        use crate::genus95_curve_code::{BaseMessage, product_code_message};

        let gpow = |r: usize| -> F128 {
            if r < 64 {
                F128 {
                    lo: 1u64 << r,
                    hi: 0,
                }
            } else {
                F128 {
                    lo: 0,
                    hi: 1u64 << (r - 64),
                }
            }
        };
        let mut rng = Rng(0xABCD_1234_0000_0001);
        let vn = 4usize;
        let mut am = vec![[0u64; 128]; vn];
        let mut bm = vec![[0u64; 128]; vn];
        let mut cm = vec![[0u64; 128]; vn];
        for o in 0..vn {
            for r in 0..128 {
                am[o][r] = rng.n();
                bm[o][r] = rng.n();
                cm[o][r] = rng.n();
            }
        }
        let eq: Vec<F128> = (0..vn)
            .map(|_| F128 {
                lo: rng.n(),
                hi: rng.n(),
            })
            .collect();
        let ap: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&am[o])).collect();
        let bp: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&bm[o])).collect();
        let cp: Vec<[uint8x16_t; 64]> = (0..vn).map(|o| bitslice(&cm[o])).collect();

        let (ab, c) = bin_abc(&ap, &bp, &cp, &eq, vn);
        let (ab_p, c_p) = bin_abc_par(&ap, &bp, &cp, &eq, vn);
        assert!(
            (0..160).all(|s| ab_p[s] == ab[s] && c_p[s] == c[s]),
            "par != serial"
        );

        let mut ab_ref = [F128::ZERO; 160];
        let mut c_ref = [F128::ZERO; 160];
        for o in 0..vn {
            for r in 0..128 {
                let w = eq[o] * gpow(r);
                let pm = product_code_message(BaseMessage(am[o][r]), BaseMessage(bm[o][r]));
                let cx = extended_base_product_message(BaseMessage(cm[o][r]));
                for s in 0..158 {
                    if pm.get_bit(64 + s) {
                        ab_ref[s] += w;
                    }
                    if cx.get_bit(64 + s) {
                        c_ref[s] += w;
                    }
                }
            }
        }
        for s in 0..158 {
            assert_eq!(ab[s], ab_ref[s], "AB fresh coord {s}");
            assert_eq!(c[s], c_ref[s], "C fresh coord {s}");
        }
    }

    /// The packed reader produces the same blocks as direct bitslicing: pack
    /// random 64-bit messages 8 bytes each (128 per 1024-byte block), and check
    /// `bin_abc_packed` equals `bin_abc` on the bitsliced messages.
    #[test]
    fn packed_reader_matches_bitsliced() {
        let mut rng = Rng(0xFEED_BEEF_0000_0001);
        let n = 5usize;
        let mut am = vec![[0u64; 128]; n];
        let mut bm = vec![[0u64; 128]; n];
        let mut cm = vec![[0u64; 128]; n];
        for o in 0..n {
            for i in 0..128 {
                am[o][i] = rng.n();
                bm[o][i] = rng.n();
                cm[o][i] = rng.n();
            }
        }
        let pack = |ms: &[[u64; 128]]| -> Vec<u8> {
            let mut p = vec![0u8; ms.len() * 1024];
            for (o, blk) in ms.iter().enumerate() {
                for i in 0..128 {
                    p[o * 1024 + i * 8..o * 1024 + i * 8 + 8]
                        .copy_from_slice(&blk[i].to_le_bytes());
                }
            }
            p
        };
        let (a_packed, b_packed, c_packed) = (pack(&am), pack(&bm), pack(&cm));
        let eq: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: rng.n(),
                hi: rng.n(),
            })
            .collect();

        let ap: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&am[o])).collect();
        let bp: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&bm[o])).collect();
        let cp: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&cm[o])).collect();

        let via_packed = bin_abc_packed(&a_packed, &b_packed, &c_packed, &eq);
        let via_blocks = bin_abc(&ap, &bp, &cp, &eq, n);
        assert!(
            (0..160)
                .all(|s| via_packed.0[s] == via_blocks.0[s] && via_packed.1[s] == via_blocks.1[s]),
            "packed reader != direct bitslice"
        );
    }

    /// The fast LUT path equals the trusted `encode_direct` path on the real
    /// fresh coords and the c message, for the same packed witness.
    #[test]
    fn round1_lut_matches_raw() {
        let mut rng = Rng(0x1234_5678);
        let n = 4usize;
        let mk = |rng: &mut Rng| -> Vec<u8> {
            let mut p = vec![0u8; n * 1024];
            for x in p.iter_mut() {
                *x = rng.n() as u8;
            }
            p
        };
        let a = mk(&mut rng);
        let b = mk(&mut rng);
        let c = mk(&mut rng);
        let eq: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: rng.n(),
                hi: rng.n(),
            })
            .collect();

        let (lut_ab, lut_w) = round1_lut_packed(&a, &b, &c, &eq);
        let (raw_ab, raw_w) = round1_raw_packed(&a, &b, &c, &eq);
        assert!(
            (0..158).all(|s| lut_ab[s] == raw_ab[s]),
            "LUT AB != raw AB on fresh coords"
        );
        assert!(
            (0..64).all(|k| lut_w[k] == raw_w[k]),
            "LUT wbar != raw wbar"
        );
    }

    /// The Paar SLP path equals the trusted `encode_direct` path — validates the
    /// generated `slp_derived::encode_slp_derived` against the derived `M`.
    #[test]
    fn round1_slp_matches_raw() {
        let mut rng = Rng(0x9ABC_DEF0);
        // Cover even n (all block-pairs) and odd n (exercises the trailing
        // single-block path via odd-length chunks).
        for n in [4usize, 3, 5, 1] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * 1024];
                for x in p.iter_mut() {
                    *x = rng.n() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.n(),
                    hi: rng.n(),
                })
                .collect();

            let (slp_ab, slp_w) = round1_slp_packed(&a, &b, &c, &eq);
            let (raw_ab, raw_w) = round1_raw_packed(&a, &b, &c, &eq);
            assert!(
                (0..158).all(|s| slp_ab[s] == raw_ab[s]),
                "SLP AB != raw AB (n={n})"
            );
            assert!(
                (0..64).all(|k| slp_w[k] == raw_w[k]),
                "SLP wbar != raw wbar (n={n})"
            );
        }
    }

    /// PROTOTYPE: the buffer-fused path ([`round1_slp_packed_fused`]) is
    /// bit-identical to [`round1_slp_packed`] on all 160 res coords + wbar.
    #[test]
    fn round1_slp_fused_matches_slp() {
        let mut rng = Rng(0xF05E_D001);
        for n in [4usize, 3, 5, 1, 16] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * 1024];
                for x in p.iter_mut() {
                    *x = rng.n() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.n(),
                    hi: rng.n(),
                })
                .collect();

            let (ab, w) = round1_slp_packed(&a, &b, &c, &eq);
            let (fab, fw) = round1_slp_packed_fused(&a, &b, &c, &eq);
            assert!(
                (0..160).all(|s| ab[s] == fab[s]),
                "fused res != res (n={n})"
            );
            assert!((0..64).all(|k| w[k] == fw[k]), "fused wbar != wbar (n={n})");
        }
    }

    /// PROTOTYPE: the fused banks path ([`round1_slp_packed_banks_fused`]) is
    /// bit-identical to [`round1_slp_packed_banks`] on res + both banks.
    #[test]
    fn round1_slp_banks_fused_matches_banks() {
        let mut rng = Rng(0xBA2C_F05E);
        for n in [4usize, 3, 5, 1, 16] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * 1024];
                for x in p.iter_mut() {
                    *x = rng.n() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.n(),
                    hi: rng.n(),
                })
                .collect();

            let (ab, b0, b1) = round1_slp_packed_banks(&a, &b, &c, &eq);
            let (fab, f0, f1) = round1_slp_packed_banks_fused(&a, &b, &c, &eq);
            assert!(
                (0..160).all(|s| ab[s] == fab[s]),
                "fused banks res != res (n={n})"
            );
            assert!(
                (0..64).all(|k| b0[k] == f0[k] && b1[k] == f1[k]),
                "fused banks != banks (n={n})"
            );
        }
    }

    /// The two-bank c-fold ([`round1_slp_packed_banks`]) reconstitutes the same
    /// AB message and the same `wbar` as [`round1_slp_packed`]: `res` identical
    /// and `bank0[k] + bank1[k] == wbar[k]` (the even/odd bit split is a partition
    /// of `pf`, so the field-mult distributes back to the original fold).
    #[test]
    fn round1_slp_banks_sum_matches_wbar() {
        let mut rng = Rng(0x5A17_BA17);
        for n in [4usize, 3, 5, 1] {
            let mk = |rng: &mut Rng| -> Vec<u8> {
                let mut p = vec![0u8; n * 1024];
                for x in p.iter_mut() {
                    *x = rng.n() as u8;
                }
                p
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            let c = mk(&mut rng);
            let eq: Vec<F128> = (0..n)
                .map(|_| F128 {
                    lo: rng.n(),
                    hi: rng.n(),
                })
                .collect();

            let (ab, w) = round1_slp_packed(&a, &b, &c, &eq);
            let (ab2, bank0, bank1) = round1_slp_packed_banks(&a, &b, &c, &eq);
            assert!((0..158).all(|s| ab[s] == ab2[s]), "banks AB != AB (n={n})");
            assert!(
                (0..64).all(|k| bank0[k] + bank1[k] == w[k]),
                "bank0 + bank1 != wbar (n={n})"
            );
        }
    }

    /// Perf probe: the derived-`M` round-1 encodes — legacy SLP (M_MASK, the perf
    /// ceiling), `encode_direct` (correctness placeholder), and the four-Russians
    /// LUT (production). Run:
    /// `cargo test --release --lib _bench_encode_paths -- --ignored --nocapture`.
    #[ignore]
    #[test]
    fn _bench_encode_paths() {
        use std::time::Instant;
        let n = 8192usize;
        let mut rng = Rng(0xBEEF);
        let mut am = vec![[0u64; 128]; n];
        let mut bm = vec![[0u64; 128]; n];
        let mut cm = vec![[0u64; 128]; n];
        for o in 0..n {
            for r in 0..128 {
                am[o][r] = rng.n();
                bm[o][r] = rng.n();
                cm[o][r] = rng.n();
            }
        }
        let ap: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&am[o])).collect();
        let bp: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&bm[o])).collect();
        let cp: Vec<[uint8x16_t; 64]> = (0..n).map(|o| bitslice(&cm[o])).collect();
        let pack = |ms: &[[u64; 128]]| -> Vec<u8> {
            let mut p = vec![0u8; ms.len() * 1024];
            for (o, blk) in ms.iter().enumerate() {
                for r in 0..128 {
                    p[o * 1024 + r * 8..o * 1024 + r * 8 + 8]
                        .copy_from_slice(&blk[r].to_le_bytes());
                }
            }
            p
        };
        let (a_packed, b_packed, c_packed) = (pack(&am), pack(&bm), pack(&cm));
        let eq: Vec<F128> = (0..n)
            .map(|_| F128 {
                lo: rng.n(),
                hi: rng.n(),
            })
            .collect();
        let nspr = |ms: f64| ms * 1e6 / (n as f64 * 128.0);

        let mut slp = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let r = bin_abc_bitslice(&ap, &bp, &cp, &eq, n);
            slp = slp.min(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&r);
        }
        let mut dir = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let r = round1_raw_packed(&a_packed, &b_packed, &c_packed, &eq);
            dir = dir.min(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&r);
        }
        let mut lut = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let r = round1_lut_packed(&a_packed, &b_packed, &c_packed, &eq);
            lut = lut.min(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&r);
        }
        let mut slpp = f64::INFINITY;
        for _ in 0..7 {
            let t = Instant::now();
            let r = round1_slp_packed(&a_packed, &b_packed, &c_packed, &eq);
            slpp = slpp.min(t.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(&r);
        }
        eprintln!("SLP pre-bitsliced (ceiling)  : {:.3} ns/row", nspr(slp));
        eprintln!("encode_direct (placeholder)  : {:.3} ns/row", nspr(dir));
        eprintln!(
            "four-Russians LUT (packed)   : {:.3} ns/row  ({:.2}x over RS 14.1)",
            nspr(lut),
            14.1 / nspr(lut)
        );
        eprintln!(
            "Paar SLP derived-M (packed)  : {:.3} ns/row  ({:.2}x over RS, {:.2}x over LUT)",
            nspr(slpp),
            14.1 / nspr(slpp),
            nspr(lut) / nspr(slpp)
        );
    }

    /// CODEGEN: Paar greedy SLP for the derived `M`, emitted to
    /// `src/genus95_curve_code/slp_derived.rs`. Run:
    /// `cargo test --release --lib _generate_slp_derived -- --ignored --nocapture`.
    #[ignore]
    #[test]
    fn _generate_slp_derived() {
        use std::collections::{BTreeSet, HashMap};
        let m = derived_m();
        // Rows over signals; signals 0..64 are the inputs. Paar repeatedly pulls
        // out the most-common co-occurring pair into a new signal (one XOR gate).
        let mut rows: Vec<BTreeSet<usize>> = (0..160)
            .map(|k| (0..64).filter(|&j| (m[k] >> j) & 1 == 1).collect())
            .collect();
        let mut gates: Vec<(usize, usize)> = Vec::new();
        let mut next = 64usize;
        loop {
            let mut counts: HashMap<(usize, usize), u32> = HashMap::new();
            for row in &rows {
                let v: Vec<usize> = row.iter().copied().collect();
                for i in 0..v.len() {
                    for j in (i + 1)..v.len() {
                        *counts.entry((v[i], v[j])).or_insert(0) += 1;
                    }
                }
            }
            // Deterministic pick: highest count, ties broken by smallest pair.
            let mut best: Option<((usize, usize), u32)> = None;
            for (&pair, &c) in &counts {
                best = match best {
                    Some((bp, bc)) if bc > c || (bc == c && bp < pair) => Some((bp, bc)),
                    _ => Some((pair, c)),
                };
            }
            let ((a, b), cnt) = match best {
                Some(x) => x,
                None => break,
            };
            if cnt < 2 {
                break;
            }
            let s = next;
            next += 1;
            gates.push((a, b));
            for row in &mut rows {
                if row.contains(&a) && row.contains(&b) {
                    row.remove(&a);
                    row.remove(&b);
                    row.insert(s);
                }
            }
        }
        let chain: usize = rows.iter().map(|r| r.len().saturating_sub(1)).sum();
        eprintln!(
            "SLP(derived M): {} gates + {} chain XORs = {} total ops",
            gates.len(),
            chain,
            gates.len() + chain
        );

        let sig = |i: usize| {
            if i < 64 {
                format!("inp[{i}]")
            } else {
                format!("s{i}")
            }
        };
        let mut src = String::new();
        src.push_str(
            "//! AUTO-GENERATED by `round1::tests::_generate_slp_derived` — Paar greedy\n",
        );
        src.push_str(
            "//! straight-line program for the evaluator-derived `M` (160x64). Do not edit.\n",
        );
        src.push_str("use std::arch::aarch64::*;\n\n");
        src.push_str("#[inline(never)]\n");
        src.push_str("pub(crate) unsafe fn encode_slp_derived(inp: &[uint8x16_t; 64], out: &mut [uint8x16_t; 160]) {\n    unsafe {\n");
        for (g, &(a, b)) in gates.iter().enumerate() {
            src.push_str(&format!(
                "        let s{} = veorq_u8({}, {});\n",
                64 + g,
                sig(a),
                sig(b)
            ));
        }
        for k in 0..160 {
            let v: Vec<usize> = rows[k].iter().copied().collect();
            if v.is_empty() {
                src.push_str(&format!("        out[{k}] = vdupq_n_u8(0);\n"));
            } else {
                let mut e = sig(v[0]);
                for &x in &v[1..] {
                    e = format!("veorq_u8({}, {})", e, sig(x));
                }
                src.push_str(&format!("        out[{k}] = {e};\n"));
            }
        }
        src.push_str("    }\n}\n");
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/genus95_curve_code/slp_derived.rs"
        );
        std::fs::write(path, src).expect("write slp_derived.rs");
        eprintln!("wrote {path}");
    }
}
