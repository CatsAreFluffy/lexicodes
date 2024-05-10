#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <x86intrin.h>
#if 0
// pretend it's icelake
__m512i mm512_permutexvar_epi8(__m512i idx, __m512i a){
    uint8_t idx2[64], a2[64];
    _mm512_storeu_epi8((void*)idx2, idx);
    _mm512_storeu_epi8((void*)a2, a);
    for(int i = 0; i < 64; i++){
        idx2[i] = a2[idx2[i] & 63];
    }
    return _mm512_loadu_epi8((void*)idx2);
}
__m512i mm512_permutex2var_epi8(__m512i a, __m512i idx, __m512i b){
    uint8_t idx2[64], a2[128];
    _mm512_storeu_epi8((void*)idx2, idx);
    _mm512_storeu_epi8((void*)a2, a);
    _mm512_storeu_epi8((void*)&a2[64], b);
    for(int i = 0; i < 64; i++){
        idx2[i] = a2[idx2[i] & 127];
    }
    return _mm512_loadu_epi8((void*)idx2);
}
__m512i mm512_gf2p8affine_epi64_epi8(__m512i x, __m512i A, int b){
    uint8_t x2[64], a2[64];
    _mm512_storeu_epi8((void*)x2, x);
    _mm512_storeu_epi8((void*)a2, A);
    for(size_t i = 0; i < 64; i++){
        uint8_t acc = 0;
        for(size_t j = 0; j < 8; j++){
            if(_popcnt32(x2[i] & a2[(i & 56) + (j ^ 7)]) & 1){
                acc |= 1 << j;
            }
        }
        x2[i] = acc ^ b;
    }
    return _mm512_loadu_epi8((void*)x2);
}
// anyway
#else
#define mm512_permutexvar_epi8 _mm512_permutexvar_epi8
#define mm512_permutex2var_epi8 _mm512_permutex2var_epi8
#define mm512_gf2p8affine_epi64_epi8 _mm512_gf2p8affine_epi64_epi8
#endif
__m512i mm512_cvtsi64_si512(uint64_t a){
    // extra vmovdqa on gcc
    // return _mm512_inserti32x4(_mm512_setzero_epi32(), _mm_cvtsi64_si128(a), 0);
    return _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0, a);
}
__m256i mm256_cvtsi64_si256(uint64_t a){
    return _mm256_set_epi64x(0, 0, 0, a);
}
__m128i mm_iota_epi8(){
	uint8_t v[16];
	for(int i = 0; i < 16; i++){
		v[i] = i;
	}
	return _mm_loadu_epi8((void*)v);
}
__m512i mm512_iota_epi8(){
	uint8_t v[64];
	for(int i = 0; i < 64; i++){
		v[i] = i;
	}
	return _mm512_loadu_epi8((void*)v);
}
__m512i mm512_iota_epi16(){
	uint16_t v[32];
	for(int i = 0; i < 32; i++){
		v[i] = i;
	}
	return _mm512_loadu_epi16((void*)v);
}
__m512i mm512_iota_epi32(){
	uint32_t v[16];
	for(int i = 0; i < 16; i++){
		v[i] = i;
	}
	return _mm512_loadu_epi32((void*)v);
}
void print(__m128i a){
    uint8_t v[16];
    _mm_storeu_si128((__m128i*)v, a);
    for(int i = 0; i < 16; i++){
        printf("%02x,", v[i]);
    }
    printf("\n");
}
__m128i x4a_decompress(uint16_t a){
    // base
    __m128i v = _mm_set1_epi8(a >> 5);
    // offset
    uint32_t offsets[8] = {0x00000000, 0x01000000, 0x01010000, 0x01000100, 0x00010100, 0x01010100, 0x02010100, 0x33221100};
    v = _mm_add_epi8(v, _mm_cvtsi32_si128(offsets[(a >> 2) & 7]));
    // refl
    v = _mm_shuffle_epi8(v, _mm_xor_si128(mm_iota_epi8(), _mm_set1_epi8(a & 3)));
    return v;
}
bool eq(__m128i a, __m128i b){
    return _mm_cvtsi128_si32(a) == _mm_cvtsi128_si32(b);
}
uint16_t x4a_compress(__m128i a){
	__m128i hmin = _mm_min_epu8(a, _mm_bsrli_si128(a, 2));
	uint8_t base = _mm_cvtsi128_si32(_mm_min_epu8(hmin, _mm_bsrli_si128(hmin, 1)));
	uint32_t diffs = _mm_cvtsi128_si32(_mm_sub_epi8(a, _mm_set1_epi8(base)));
	uint32_t offsets[8] = {0x00000000, 0x01000000, 0x01010000, 0x01000100, 0x00010100, 0x01010100, 0x02010100, 0x33221100};
	uint32_t offsets2[32];
	for(int i = 0; i < 8; i++){
		uint32_t x = offsets[i];
		offsets2[i*4] = x;
		offsets2[i*4+1] = ((x >> 8) & 0x00ff00ff) | ((x << 8) & 0xff00ff00);
		offsets2[i*4+2] = (x >> 16) | (x << 16);
		offsets2[i*4+3] = _bswap(x);
	}
	for(int i = 0; i < 32; i++){
		if(diffs == offsets2[i]){
			return (base << 5) + i;
		}
	}
	// sussy
	return 255;
}
void print(__m512i a){
    // uint16_t v[32];
    // _mm512_storeu_epi16((void*)v, a);
    // for(int i = 0; i < 32; i++){
    //     printf("%04x,", v[i]);
    // }
    // printf("\n");
    uint8_t v[64];
    _mm512_storeu_epi8((void*)v, a);
    for(int i = 0; i < 64; i++){
        printf("%02x,", v[i]);
    }
    printf("\n");
}
void print2(__m512i a){
    print(a);
}
void bake(__m512i a){
	uint64_t a2[8];
	_mm512_storeu_epi16((void*)a2, a);
	printf("_mm512_setr_epi64(");
	for(int i = 0; i < 8; i++){
		printf("0x%016lx%s", a2[i], i < 7 ? ", " : ")\n");
	}
}

class x4a_icx_zmm_block {
    // move this later
    public:
    static __m512i permdec(__m512i a, uint16_t gen){
        // printf("permdec input gen %04x a\n", gen);
        // printzmm(a);
        // a = _mm512_permutexvar_epi32(_mm512_xor_si512(mm512_iota_epi32(), _mm512_set1_epi32(gen >> 4)), a);
        // // printf("1\n");
        // // printzmm(a);
        // a = _mm512_shuffle_epi8(a, _mm512_xor_si512(_mm512_and_si512(mm512_iota_epi8(), _mm512_set1_epi8(15)), _mm512_set1_epi8((gen >> 2) & 3)));
        // 4 entries per byte
        a = mm512_permutexvar_epi8(_mm512_xor_si512(mm512_iota_epi8(), _mm512_set1_epi8(gen >> 2)), a);
        // printf("2\n");
        // printzmm(a);
        // words with base >0
        __mmask64 big = _mm512_cmpge_epu8_mask(a, _mm512_set1_epi8(32));
        // permute within words
        a = _mm512_xor_si512(a, _mm512_set1_epi8(gen & 3));
        // printf("3\n");
        // printzmm(a);
        // words with base 0 need to be decreased by 20 (offset 6->1, others->0)
        __m512i a2 = _mm512_subs_epu8(a, _mm512_set1_epi8(20));
        // printf("4\n");
        // printzmm(a2);
        // other words need to be decreased by 32
        a2 = _mm512_mask_sub_epi8(a2, big, a, _mm512_set1_epi8(32));
        // printf("permdec output\n");
        // printzmm(a2);
        return a2;
    }
    static __m512i gen_max_table(int block){
        uint8_t table[64];
        for(int i = 0; i < 64; i++){
            int j = (block << 6) + i;
            uint8_t lo = (j >> 3) & 28;
            uint8_t hi = (j & 31) ^ lo;
            if(hi < lo){
                hi ^= 28;
                lo ^= 28;
            }
            hi -= 4;
            table[i] = x4a_compress(_mm_max_epu8(x4a_decompress(hi), x4a_decompress(lo)));
        }
        return _mm512_loadu_epi8((void*)table);
    }
    static __m512i lookup(__m512i a){
        // on icelake this is just a vpermi2b
        // outputs of gen_max_table()
        __m512i table[2] = {
            _mm512_setr_epi64(0x00000000fffefdfc, 0x0a0a080807060504, 0x101111100d0c0d0c, 0x1b1a191817161514, 0x312c281804040404, 0x110c080416160808, 0x150c150c14111114, 0x20161514271a1918),
            _mm512_setr_epi64(0x2020080815140808, 0x2020201427262518, 0x2726191820201514, 0x1415151415141514, 0x1020201016161414, 0x1b26251817202014, 0x20162014271a2518, 0x200c200c14161614)
        };
        return mm512_permutex2var_epi8(table[0], a, table[1]);
    }
    static __m512i max(__m512i a, __m512i b){
        __m512i hi = _mm512_max_epu8(a, b);
        __m512i lo = _mm512_min_epu8(a, b);
        __m512i lobase = _mm512_and_si512(lo, _mm512_set1_epi8(0xe0));
        __m512i hibase = _mm512_and_si512(hi, _mm512_set1_epi8(0xe0));
        __m512i diff = _mm512_sub_epi8(hi, lobase);
        __mmask64 nkeeps = _mm512_cmplt_epu8_mask(diff, _mm512_set1_epi8(64));
        __mmask64 neqs = _mm512_cmpge_epu8_mask(diff, _mm512_set1_epi8(32));
        __m512i hi2 = _mm512_add_epi8(hi, _mm512_set1_epi8(0b00000100));
        // could also be an and/subs
        // __m512i decbase = _mm512_subs_epu8(_mm512_and_si512(mm512_iota_epi8(), _mm512_set1_epi8(31)), _mm512_set1_epi8(20));
        // __m512i lo2 = _mm512_mask_permutexvar_epi8(lo, neqs, lo, decbase);
        __m512i lo2 = _mm512_mask_subs_epu8(lo, neqs, _mm512_and_si512(lo, _mm512_set1_epi8(31)), _mm512_set1_epi8(20));
        // copy offset to base, then xor bits of base with the top bit
        __m512i lo3 = mm512_gf2p8affine_epi64_epi8(lo2, _mm512_set1_epi64(0x0102040810141800ull), 0);
        // __m512i idxs = _mm512_xor_si512(hi2, lo2);
        // // c?b:a
        // idxs = _mm512_ternarylogic_epi32(idxs, _mm512_slli_epi16(lo2, 3), _mm512_set1_epi16(0x00e0), 0b11011000);
        // c?a^b:a
        __m512i idxs = _mm512_ternarylogic_epi32(lo3, hi2, _mm512_set1_epi8(0b11111), 0b01111000);
        __m512i r = lookup(idxs);
        // fix rolls
        // c?a^b:a
        r = _mm512_ternarylogic_epi32(r, lo2, _mm512_set1_epi8(0b11), 0b01111000);
        // r = _mm512_add_epi16(r, hibase);
        // r = _mm512_mask_blend_epi16(keeps, hi, r);
        return _mm512_mask_add_epi8(hi, nkeeps, r, hibase);
    }
    // static __m512i gen_zpos_table(){
    //     uint16_t table[32];
    //     for(int i = 0; i < 32; i++){
    //         uint8_t v[16];
    //         _mm_storeu_epi8((void*)v, x4a_decompress(i));
    //         table[i] = 128;
    //         for(int j = 0; j < 16; j++){
    //             if(v[j] == 0){
    //                 table[i] = j;
    //                 break;
    //             }
    //         }
    //     }
    //     return _mm512_loadu_epi16((void*)table);
    // }
    static uint16_t zeropos_(__m512i a){
        // too lazy to do an elegant algorithm
        uint8_t v[64];
        _mm512_storeu_epi8((void*)v, a);
        for(int i = 0; i < 64; i++){
            uint8_t v2[16];
            _mm_storeu_epi8((void*)v2, x4a_decompress(v[i]));
            for(int j = 0; j < 4; j++){
                if(!v2[j]){
                    return i * 4 + j;
                }
            }
        }
        return 256;
        // // printzmm(a);
        // __mmask32 haszero = _mm512_cmplt_epu16_mask(a, _mm512_set1_epi16(32));
        // // output of gen_zpos_table
        // // __m512i table = gen_zpos_table();
        // __m512i table = _mm512_setr_epi64(0x0000000000000000, 0x0001000000000000, 0x0002000200000000, 0x0001000000010000, 0x0000000100010000, 0x0003000200010000, 0x0003000200010000, 0x0003000200010000);
        // __m512i positionsz = _mm512_mask_permutexvar_epi16(_mm512_set1_epi16(0x0100), haszero, a, table);
        // positionsz = _mm512_add_epi16(positionsz, _mm512_slli_epi16(mm512_iota_epi16(), 2));
        // // print(positionsz);
        // __m256i positionsy = _mm256_min_epu16(_mm512_extracti32x8_epi32(positionsz, 0), _mm512_extracti32x8_epi32(positionsz, 1));
        // __m128i positionsx = _mm_min_epu16(_mm256_extracti128_si256(positionsy, 0), _mm256_extracti128_si256(positionsy, 1));
        // // print(positionsx);
        // // wonderful instruction
        // return _mm_cvtsi128_si32(_mm_minpos_epu16(positionsx));
    }
    __m512i data;
    x4a_icx_zmm_block(__m512i _data){
        data = _data;
    }
    static constexpr uint32_t entries_per_block_bits = 8;
    static x4a_icx_zmm_block base(uint32_t dist){
        return x4a_icx_zmm_block(mm512_cvtsi64_si512(((dist - 3) << 5) | 27));
    }
    static std::vector<uint64_t> base_gens(uint32_t dist){
        std::vector<uint64_t> v = {1, 2, 4};
        return v;
    }
    x4a_icx_zmm_block combine(x4a_icx_zmm_block other, uint64_t gen){
        __m512i a = data;
        __m512i b = other.data;
        b = permdec(b, gen);
        return x4a_icx_zmm_block(max(a, b));
    }
    bool haszero(){
        return _mm512_cmplt_epu8_mask(data, _mm512_set1_epi8(32)) != (__mmask64)0;
    }
    uint64_t zeropos(){
        return zeropos_(data);
    }
    void print(){
        uint8_t v[64];
        _mm512_storeu_si512((void*)v, data);
        for(int i = 0; i < 64; i++){
            printf("%08x,", _mm_cvtsi128_si32(x4a_decompress(v[i])));
        }
        printf("\n");
        for(int i = 0; i < 64; i++){
            printf("%02x,", v[i]);
        }
        printf("\n");
    }
};

class x4a_skx_zmm_block {
    // move this later
    public:
    static __m512i permdec(__m512i a, uint16_t gen){
        // printf("permdec input gen %04x a\n", gen);
        // printzmm(a);
        // 4 entries per byte, 4 bytes per dword, so /16
        a = _mm512_permutexvar_epi32(_mm512_xor_si512(mm512_iota_epi32(), _mm512_set1_epi32(gen >> 4)), a);
        // printf("1\n");
        // printzmm(a);
        a = _mm512_shuffle_epi8(a, _mm512_xor_si512(_mm512_and_si512(mm512_iota_epi8(), _mm512_set1_epi8(15)), _mm512_set1_epi8((gen >> 2) & 3)));
        // printf("2\n");
        // printzmm(a);
        // words with base >0
        __mmask64 big = _mm512_cmpge_epu8_mask(a, _mm512_set1_epi8(32));
        // permute within words
        a = _mm512_xor_si512(a, _mm512_set1_epi8(gen & 3));
        // printf("3\n");
        // printzmm(a);
        // words with base 0 need to be decreased by 20 (offset 6->1, others->0)
        __m512i a2 = _mm512_subs_epu8(a, _mm512_set1_epi8(20));
        // printf("4\n");
        // printzmm(a2);
        // other words need to be decreased by 32
        a2 = _mm512_mask_subs_epu8(a2, big, a, _mm512_set1_epi8(32));
        // printf("permdec output\n");
        // printzmm(a2);
        return a2;
    }
    static __m512i gen_max_table(int block){
        uint16_t table[32];
        for(int i = 0; i < 32; i++){
            int j = (block << 5) + i;
            uint8_t lo = (j >> 3) & 28;
            uint8_t hi = (j & 31) ^ lo;
            if(hi < lo){
                hi ^= 28;
                lo ^= 28;
            }
            hi -= 4;
            table[i] = x4a_compress(_mm_max_epu8(x4a_decompress(hi), x4a_decompress(lo)));
        }
        return _mm512_loadu_epi16((void*)table);
    }
    static __m512i lookup(__m512i a){
        // on icelake this'd just be a vpermi2b
        // outputs of gen_max_table()
        __m512i table[4] = {
            _mm512_setr_epi64(0x00ff00fe00fd00fc, 0x0000000000000000, 0x0007000600050004, 0x000a000a00080008, 0x000d000c000d000c, 0x0010001100110010, 0x0017001600150014, 0x001b001a00190018),
            _mm512_setr_epi64(0x0004000400040004, 0x0031002c00280018, 0x0016001600080008, 0x0011000c00080004, 0x0014001100110014, 0x0015000c0015000c, 0x0027001a00190018, 0x0020001600150014),
            _mm512_setr_epi64(0x0015001400080008, 0x0020002000080008, 0x0027002600250018, 0x0020002000200014, 0x0020002000150014, 0x0027002600190018, 0x0015001400150014, 0x0014001500150014),
            _mm512_setr_epi64(0x0016001600140014, 0x0010002000200010, 0x0017002000200014, 0x001b002600250018, 0x0027001a00250018, 0x0020001600200014, 0x0014001600160014, 0x0020000c0020000c)
        };
        // move low bytes of words to high bytes
        __m512i table2[4];
        for(int i = 0; i < 4; i++){
            table2[i] = _mm512_slli_epi16(table[i], 8);
        }
        __mmask64 later = _mm512_cmpge_epu8_mask(a, _mm512_set1_epi8(64));
        __m512i early1 = _mm512_permutex2var_epi16(table[0], a, table[1]);
        __m512i late1 = _mm512_permutex2var_epi16(table[2], a, table[3]);
        __m512i a2 = _mm512_srli_epi16(a, 8);
        __m512i early2 = _mm512_permutex2var_epi16(table2[0], a2, table2[1]);
        __m512i late2 = _mm512_permutex2var_epi16(table2[2], a2, table2[3]);
        __m512i early = _mm512_add_epi8(early1, early2);
        __m512i late = _mm512_add_epi8(late1, late2);
        return _mm512_mask_blend_epi8(later, early, late);
    }
    static __m512i max(__m512i a, __m512i b){
        __m512i hi = _mm512_max_epu8(a, b);
        __m512i lo = _mm512_min_epu8(a, b);
        __m512i lobase = _mm512_and_si512(lo, _mm512_set1_epi8(0xe0));
        __m512i hibase = _mm512_and_si512(hi, _mm512_set1_epi8(0xe0));
        __m512i diff = _mm512_sub_epi8(hi, lobase);
        __mmask64 nkeeps = _mm512_cmplt_epu8_mask(diff, _mm512_set1_epi8(64));
        __mmask64 neqs = _mm512_cmpge_epu8_mask(diff, _mm512_set1_epi8(32));
        __m512i hi2 = _mm512_add_epi8(hi, _mm512_set1_epi8(0b00000100));
        // __m512i decbase = _mm512_subs_epu8(mm512_iota_epi8(), _mm512_set1_epi8(20));
        // could also be an and/subs
        // __m512i lo2 = _mm512_mask_permutexvar_epi16(lo, neqs, lo, decbase);
        __m512i lo2 = _mm512_mask_subs_epu8(lo, neqs, _mm512_and_si512(lo, _mm512_set1_epi8(31)), _mm512_set1_epi8(20));
        __m512i idxs = _mm512_xor_si512(hi2, lo2);
        // replace base with lo2.offset
        // todo: use gfni for icelake
        // c?b:a
        idxs = _mm512_ternarylogic_epi32(idxs, _mm512_slli_epi16(lo2, 3), _mm512_set1_epi8(0xe0), 0b11011000);
        __mmask64 packs = _mm512_test_epi8_mask(lo2, _mm512_set1_epi8(16));
        __m512i flips = _mm512_maskz_set1_epi8(packs, 0b11100000);
        // Toggle high bits if lo.offset>=4 (which doesn't lose information since this changes
        // hi>=lo-1 to hi<lo-1, note that hi.offset=0 and lo.offset=1 is possible)
        idxs = _mm512_xor_si512(idxs, flips);
        __m512i r = lookup(idxs);
        // fix rolls
        // c?a^b:a
        r = _mm512_ternarylogic_epi32(r, lo2, _mm512_set1_epi8(0b11), 0b01111000);
        // r = _mm512_add_epi16(r, hibase);
        // r = _mm512_mask_blend_epi16(keeps, hi, r);
        return _mm512_mask_add_epi8(hi, nkeeps, r, hibase);
    }
    // static __m512i gen_zpos_table(){
    //     uint16_t table[32];
    //     for(int i = 0; i < 32; i++){
    //         uint8_t v[16];
    //         _mm_storeu_epi8((void*)v, x4a_decompress(i));
    //         table[i] = 128;
    //         for(int j = 0; j < 16; j++){
    //             if(v[j] == 0){
    //                 table[i] = j;
    //                 break;
    //             }
    //         }
    //     }
    //     return _mm512_loadu_epi16((void*)table);
    // }
    static uint16_t zeropos_(__m512i a){
        // too lazy to do an elegant algorithm
        uint8_t v[64];
        _mm512_storeu_epi8((void*)v, a);
        for(int i = 0; i < 64; i++){
            uint8_t v2[16];
            _mm_storeu_epi8((void*)v2, x4a_decompress(v[i]));
            for(int j = 0; j < 4; j++){
                if(!v2[j]){
                    return i * 4 + j;
                }
            }
        }
        return 256;
        // // printzmm(a);
        // __mmask32 haszero = _mm512_cmplt_epu16_mask(a, _mm512_set1_epi16(32));
        // // output of gen_zpos_table
        // // __m512i table = gen_zpos_table();
        // __m512i table = _mm512_setr_epi64(0x0000000000000000, 0x0001000000000000, 0x0002000200000000, 0x0001000000010000, 0x0000000100010000, 0x0003000200010000, 0x0003000200010000, 0x0003000200010000);
        // __m512i positionsz = _mm512_mask_permutexvar_epi16(_mm512_set1_epi16(0x0100), haszero, a, table);
        // positionsz = _mm512_add_epi16(positionsz, _mm512_slli_epi16(mm512_iota_epi16(), 2));
        // // print(positionsz);
        // __m256i positionsy = _mm256_min_epu16(_mm512_extracti32x8_epi32(positionsz, 0), _mm512_extracti32x8_epi32(positionsz, 1));
        // __m128i positionsx = _mm_min_epu16(_mm256_extracti128_si256(positionsy, 0), _mm256_extracti128_si256(positionsy, 1));
        // // print(positionsx);
        // // wonderful instruction
        // return _mm_cvtsi128_si32(_mm_minpos_epu16(positionsx));
    }
    __m512i data;
    x4a_skx_zmm_block(__m512i _data){
        data = _data;
    }
    static constexpr uint32_t entries_per_block_bits = 8;
    static x4a_skx_zmm_block base(uint32_t dist){
        return x4a_skx_zmm_block(mm512_cvtsi64_si512(((dist - 3) << 5) | 27));
    }
    static std::vector<uint64_t> base_gens(uint32_t dist){
        std::vector<uint64_t> v = {1, 2, 4};
        return v;
    }
    x4a_skx_zmm_block combine(x4a_skx_zmm_block other, uint64_t gen){
        __m512i a = data;
        __m512i b = other.data;
        b = permdec(b, gen);
        return x4a_skx_zmm_block(max(a, b));
    }
    bool haszero(){
        return _mm512_cmplt_epu8_mask(data, _mm512_set1_epi8(32)) != (__mmask64)0;
    }
    uint64_t zeropos(){
        return zeropos_(data);
    }
    void print(){
        uint8_t v[64];
        _mm512_storeu_si512((void*)v, data);
        for(int i = 0; i < 64; i++){
            printf("%08x,", _mm_cvtsi128_si32(x4a_decompress(v[i])));
        }
        printf("\n");
        for(int i = 0; i < 64; i++){
            printf("%02x,", v[i]);
        }
        printf("\n");
    }
};

class x4a_skx_ymm_block {
    // move this later
    public:
    static void printzmm(__m512i a){
        uint16_t v[32];
        _mm512_storeu_epi16((void*)v, a);
        for(int i = 0; i < 32; i++){
            printf("%08x,", _mm_cvtsi128_si32(x4a_decompress(v[i])));
        }
        printf("\n");
        print2(a);
    }
    // internal representation is as __m512i of words
    static __m512i permdec(__m512i a, uint16_t gen){
        // printf("permdec input gen %04x a\n", gen);
        // printzmm(a);
        // each word is 4 entries, so permute using gen/4
        a = _mm512_permutexvar_epi16(_mm512_xor_si512(mm512_iota_epi16(), _mm512_set1_epi16(gen >> 2)), a);
        // printf("1\n");
        // printzmm(a);
        // words with base >0
        __mmask32 big = _mm512_cmpge_epu16_mask(a, _mm512_set1_epi16(32));
        // permute within words
        a = _mm512_xor_si512(a, _mm512_set1_epi16(gen & 3));
        // printf("2\n");
        // printzmm(a);
        // words with base 0 need to be decreased by 20 (offset 6->1, others->0)
        __m512i a2 = _mm512_subs_epu16(a, _mm512_set1_epi16(20));
        // printf("3\n");
        // printzmm(a2);
        // other words need to be decreased by 32
        a2 = _mm512_mask_subs_epu16(a2, big, a, _mm512_set1_epi16(32));
        // printf("permdec output\n");
        // printzmm(a2);
        return a2;
    }
    static __m512i gen_max_table(int block){
        uint16_t table[32];
        for(int i = 0; i < 32; i++){
            int j = (block<< 5) + i;
            uint8_t lo = (j >> 3) & 28;
            uint8_t hi = (j & 31) ^ lo;
            if(hi < lo){
                hi ^= 28;
                lo ^= 28;
            }
            hi -= 4;
            table[i] = x4a_compress(_mm_max_epu8(x4a_decompress(hi), x4a_decompress(lo)));
        }
        return _mm512_loadu_epi16((void*)table);
    }
    static __m512i lookup(__m512i a){
        // outputs of gen_max_table()
        __m512i table[4] = {
            _mm512_setr_epi64(0x00ff00fe00fd00fc, 0x0000000000000000, 0x0007000600050004, 0x000a000a00080008, 0x000d000c000d000c, 0x0010001100110010, 0x0017001600150014, 0x001b001a00190018),
            _mm512_setr_epi64(0x0004000400040004, 0x0031002c00280018, 0x0016001600080008, 0x0011000c00080004, 0x0014001100110014, 0x0015000c0015000c, 0x0027001a00190018, 0x0020001600150014),
            _mm512_setr_epi64(0x0015001400080008, 0x0020002000080008, 0x0027002600250018, 0x0020002000200014, 0x0020002000150014, 0x0027002600190018, 0x0015001400150014, 0x0014001500150014),
            _mm512_setr_epi64(0x0016001600140014, 0x0010002000200010, 0x0017002000200014, 0x001b002600250018, 0x0027001a00250018, 0x0020001600200014, 0x0014001600160014, 0x0020000c0020000c)
        };
        __mmask32 later = _mm512_cmpge_epu16_mask(a, _mm512_set1_epi16(64));
        __m512i early = _mm512_permutex2var_epi16(table[0], a, table[1]);
        __m512i late = _mm512_permutex2var_epi16(table[2], a, table[3]);
        return _mm512_mask_blend_epi16(later, early, late);
    }
    // static __m512i susmax(__m512i a, __m512i b){
    //     uint16_t a2[32], b2[32];
    //     _mm512_storeu_epi16((void*)a2, a);
    //     _mm512_storeu_epi16((void*)b2, b);
    //     for(int i = 0; i < 32; i++){
    //         a2[i] = x4a_compress(_mm_max_epu8(x4a_decompress(a2[i]), x4a_decompress(b2[i])));
    //     }
    //     return _mm512_loadu_epi16((void*)a2);
    // }
    static __m512i max(__m512i a, __m512i b){
        __m512i hi = _mm512_max_epu16(a, b);
        __m512i lo = _mm512_min_epu16(a, b);
        __m512i lobase = _mm512_and_si512(lo, _mm512_set1_epi16(0xffe0));
        __m512i hibase = _mm512_and_si512(hi, _mm512_set1_epi16(0xffe0));
        __m512i diff = _mm512_sub_epi16(hi, lobase);
        __mmask32 nkeeps = _mm512_cmplt_epu16_mask(diff, _mm512_set1_epi16(64));
        __mmask32 neqs = _mm512_cmpge_epu16_mask(diff, _mm512_set1_epi16(32));
        __m512i hi2 = _mm512_add_epi16(hi, _mm512_set1_epi16(0b00000100));
        __m512i decbase = _mm512_subs_epu16(mm512_iota_epi16(), _mm512_set1_epi16(20));
        // could also be an and/subs
        __m512i lo2 = _mm512_mask_permutexvar_epi16(lo, neqs, lo, decbase);
        __m512i idxs = _mm512_xor_si512(hi2, lo2);
        // replace base with lo2.offset
        // todo: use gfni for icelake
        // c?b:a
        idxs = _mm512_ternarylogic_epi32(idxs, _mm512_slli_epi16(lo2, 3), _mm512_set1_epi16(0x00e0), 0b11011000);
        __mmask32 packs = _mm512_test_epi16_mask(lo2, _mm512_set1_epi16(16));
        __m512i flips = _mm512_maskz_set1_epi16(packs, 0b11100000);
        // Toggle high bits if lo.offset>=4 (which doesn't lose information since this changes
        // hi>=lo-1 to hi<lo-1, note that hi.offset=0 and lo.offset=1 is possible)
        idxs = _mm512_xor_si512(idxs, flips);
        __m512i r = lookup(idxs);
        // fix rolls
        // c?a^b:a
        r = _mm512_ternarylogic_epi32(r, lo2, _mm512_set1_epi16(0b11), 0b01111000);
        // r = _mm512_add_epi16(r, hibase);
        // r = _mm512_mask_blend_epi16(keeps, hi, r);
        // for(int i=0;i<10;i++)r = _mm512_ternarylogic_epi32(r, lo2, _mm512_set1_epi16(0b11), 0b01111000);
        return _mm512_mask_add_epi16(hi, nkeeps, r, hibase);
    }
    static __m512i gen_zpos_table(){
        uint16_t table[32];
        for(int i = 0; i < 32; i++){
            uint8_t v[16];
            _mm_storeu_epi8((void*)v, x4a_decompress(i));
            table[i] = 128;
            for(int j = 0; j < 16; j++){
                if(v[j] == 0){
                    table[i] = j;
                    break;
                }
            }
        }
        return _mm512_loadu_epi16((void*)table);
    }
    static uint16_t zeropos_(__m512i a){
        // printzmm(a);
        __mmask32 haszero = _mm512_cmplt_epu16_mask(a, _mm512_set1_epi16(32));
        // output of gen_zpos_table
        // __m512i table = gen_zpos_table();
        __m512i table = _mm512_setr_epi64(0x0000000000000000, 0x0001000000000000, 0x0002000200000000, 0x0001000000010000, 0x0000000100010000, 0x0003000200010000, 0x0003000200010000, 0x0003000200010000);
        __m512i positionsz = _mm512_mask_permutexvar_epi16(_mm512_set1_epi16(0x0100), haszero, a, table);
        positionsz = _mm512_add_epi16(positionsz, _mm512_slli_epi16(mm512_iota_epi16(), 2));
        // print(positionsz);
        __m256i positionsy = _mm256_min_epu16(_mm512_extracti32x8_epi32(positionsz, 0), _mm512_extracti32x8_epi32(positionsz, 1));
        __m128i positionsx = _mm_min_epu16(_mm256_extracti128_si256(positionsy, 0), _mm256_extracti128_si256(positionsy, 1));
        // print(positionsx);
        // wonderful instruction
        return _mm_cvtsi128_si32(_mm_minpos_epu16(positionsx));
    }
    __m256i data;
    x4a_skx_ymm_block(__m256i _data){
        data = _data;
    }
    static constexpr uint32_t entries_per_block_bits = 7;
    static x4a_skx_ymm_block base(uint32_t dist){
        return x4a_skx_ymm_block(mm256_cvtsi64_si256(((dist - 3) << 5) | 27));
    }
    static std::vector<uint64_t> base_gens(uint32_t dist){
        std::vector<uint64_t> v = {1, 2, 4};
        return v;
    }
    x4a_skx_ymm_block combine(x4a_skx_ymm_block other, uint64_t gen){
        __m512i a = _mm512_cvtepi8_epi16(data);
        __m512i b = _mm512_cvtepi8_epi16(other.data);
        b = permdec(b, gen);
        return x4a_skx_ymm_block(_mm512_cvtepi16_epi8(max(a, b)));
    }
    bool haszero(){
        return _mm256_cmplt_epu8_mask(data, _mm256_set1_epi8(32)) != (__mmask32)0;
    }
    uint64_t zeropos(){
        return zeropos_(_mm512_cvtepu8_epi16(data));
    }
    void print(){
        uint8_t v[32];
        _mm256_storeu_si256((__m256i*)v, data);
        for(int i = 0; i < 32; i++){
            printf("%08x,", _mm_cvtsi128_si32(x4a_decompress(v[i])));
        }
        printf("\n");
        for(int i = 0; i < 32; i++){
            printf("%02x,", v[i]);
        }
        printf("\n");
    }
};

template<typename block> block first_block(uint32_t dist, std::vector<uint64_t> &gens){
    block b = block::base(dist);
    // b.print();
    gens = block::base_gens(dist);
    uint64_t gen = gens.back();
    while(gen < (1 << block::entries_per_block_bits)){
        b = b.combine(b, gen);
        // printf("gen %04lx\n", gen);
        // b.print();
        uint64_t last_gen = gen;
        if(b.haszero()){
            gen = b.zeropos();
        }else{
            gen = (1 << block::entries_per_block_bits);
        }
        gens.push_back(gen);
        if(last_gen >= gen){
            printf("sussy 1 %02lx %02lx\n", last_gen, gen);
            return b;
        }
    }
    return b;
}

template<typename block> uint64_t slice(uint32_t dist, uint64_t gen, uint64_t slice_blocks, block* lows, uint64_t low_base, block* highs, uint64_t high_base){
    uint64_t offset = gen ^ low_base ^ high_base;
    uint64_t block_offset = offset >> block::entries_per_block_bits;
    if(block_offset >= slice_blocks){
        printf("sussy 3: %d %08lx %08lx %08lx %08lx %08lx\n", dist, gen, slice_blocks, low_base, high_base, block_offset);
    }
    bool needzero = true;
    uint64_t next_gen = std::numeric_limits<uint64_t>::max();
    for(size_t i = 0; i < slice_blocks; i++){
        block lo = lows[i ^ block_offset];
        block hi = highs[i];
        // printf("i %d gen %08x\n", i, gen);
        // lo.print();
        // hi.print();
        lo = lo.combine(hi, offset);
        hi = hi.combine(lo, offset);
        // lo.print();
        // hi.print();
        lows[i ^ block_offset] = lo;
        highs[i] = hi;
        // lo.store(&lows[i ^ block_offset]);
        // hi.store(&highs[i]);
        // printf("haszero %08x %d\n", high_base + (i << block::entries_per_block_bits), hi.haszero());
        if(needzero && hi.haszero()){
            // printf("zerpos %08x %08x %08x %08x\n", high_base, (i << block::entries_per_block_bits), hi.zeropos(), high_base + (i << block::entries_per_block_bits) + hi.zeropos());
            next_gen = high_base + (i << block::entries_per_block_bits) + hi.zeropos();
            needzero = false;
        }
    }
    return next_gen;
}

template<typename block> std::vector<uint64_t> findcode(uint32_t dist, uint32_t codim){
    std::vector<uint64_t> gens;
    block t = first_block<block>(dist, gens);
    for(size_t i = 1; i < gens.size(); i++){
        printf("i %05zu v %016lx\n", i, gens[i]);
        fflush(stdout);
    }
    if(block::entries_per_block_bits >= codim){
        return gens;
    }
    // todo: deal with non-multiple-of-4k stuff correctly
    block* area = (block*)mmap(0, sizeof(block) << (codim - block::entries_per_block_bits), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE|MAP_HUGETLB, -1, 0);
    area[0] = t;
    uint64_t gen = (1 << block::entries_per_block_bits);
    while(gen < (1ull << codim)){
        uint64_t high_base = (1ull << 63) >> _lzcnt_u64(gen);
        uint64_t high_base_blocks = high_base >> block::entries_per_block_bits;
        uint64_t last_gen = gen;
        gen = std::min(slice<block>(dist, gen, high_base_blocks, area, 0, &area[high_base_blocks], high_base), high_base << 1);
        gens.push_back(gen);
        printf("i %05lu v %016lx\n", gens.size() - 1, gen);
        fflush(stdout);
        if(last_gen >= gen){
            printf("sussy 2 %08lx %08lx\n", last_gen, gen);
            return gens;
        }
    }
    munmap((void*)area, sizeof(block) << (codim - block::entries_per_block_bits));
    return gens;
}

template<typename block> void findcode_thread(uint32_t dist, uint64_t gen, block* area, uint64_t blocks, std::atomic_uint64_t* next_slice, std::atomic_uint64_t* next_gen){
    uint64_t high_base = (1ull << 63) >> _lzcnt_u64(gen);
    uint64_t high_base_blocks = high_base >> block::entries_per_block_bits;
    uint64_t slices = 0;
    uint64_t this_gen = std::numeric_limits<uint64_t>::max();
    for(;;){
        uint64_t this_slice = next_slice->fetch_add(1, std::memory_order_relaxed);
        // printf("thread %08lx %ld\n", gen, this_slice);
        // printf("%08x %d\n", gen, this_slice);
        uint64_t this_slice_blocks = this_slice * blocks;
        uint64_t this_slice_entries = this_slice_blocks << block::entries_per_block_bits;
        if(this_slice_blocks >= high_base_blocks){
            // 5000
            // if(gen == 0x2d0a0c29){
            if((gen & (gen - 1)) == 0){
                // printf("Thread slices: %lu\n", slices);
            }
            // atomic_min plz
            uint64_t cur_next_gen = next_gen->load(std::memory_order_relaxed);
            while(cur_next_gen > this_gen){
                next_gen->compare_exchange_weak(cur_next_gen, this_gen, std::memory_order_relaxed, std::memory_order_relaxed);
            }
            return;
        }
        uint64_t highs_index = high_base_blocks + this_slice_blocks;
        uint64_t lows_index = ((gen >> block::entries_per_block_bits) ^ highs_index) & (-blocks);
        // printf("thread %08lx %ld h %08lx l %08lx\n", gen, this_slice, highs_index, lows_index);
        // uint64_t this_gen = slice<block>(dist, gen, blocks, &area[lows_index], lows_index << block::entries_per_block_bits, &area[highs_index], highs_index << block::entries_per_block_bits);
        this_gen = std::min(this_gen, slice<block>(dist, gen, blocks, &area[lows_index], lows_index << block::entries_per_block_bits, &area[highs_index], highs_index << block::entries_per_block_bits));
        slices++;
    }
}

template<typename block, class blah> void findcode_thread2(uint32_t dist, uint32_t codim, uint64_t* gen, std::barrier<blah> sync_point, block* area, uint64_t blocks_per_slice, std::atomic_uint64_t* next_slice, std::atomic_uint64_t* next_gen){
    while(*gen < (1ull << codim)){
        uint64_t local_gen = *gen;
        uint64_t high_base_entry = (1ull << 63) >> _lzcnt_u64(local_gen);
        uint64_t high_base_block = high_base_entry >> block::entries_per_block_bits;
        uint64_t slices = 0;
        uint64_t local_next_gen = std::numeric_limits<uint64_t>::max();
        for(;;){
            uint64_t this_slice = next_slice->fetch_add(1, std::memory_order_relaxed);
            uint64_t slice_first_block = this_slice * blocks_per_slice;
            uint64_t slice_first_entry = slice_first_block >> block::entries_per_block_bits;
            if(slice_first_block >= high_base_block){
                break;
            }
            uint64_t high_block_index = high_base_block + slice_first_block;
            uint64_t low_block_index = ((local_gen >> block::entries_per_block_bits) ^ high_block_index) & (-blocks_per_slice);
            local_next_gen = std::min(local_next_gen, slice<block>(dist, local_gen, blocks_per_slice, &area[low_block_index], low_block_index << block::entries_per_block_bits, &area[high_block_index], high_block_index << block::entries_per_block_bits));
            slices++;
        }
        // atomic_min plz
        uint64_t cur_next_gen = next_gen->load(std::memory_order_relaxed);
        while(cur_next_gen > local_next_gen){
            next_gen->compare_exchange_weak(cur_next_gen, local_next_gen, std::memory_order_relaxed, std::memory_order_relaxed);
        }
        sync_point.arrive_and_wait();
    }
}

template<typename block> std::vector<uint64_t> findcode2(uint32_t dist, uint32_t codim, uint64_t blocks_per_slice, uint64_t max_threads){
    std::vector<uint64_t> gens;
    block t = first_block<block>(dist, gens);
    for(size_t i = 1; i < gens.size(); i++){
        // printf("i %05zu v %016lx\n", i, gens[i]);
        fflush(stdout);
    }
    // todo: remove some gens if codim<epbb
    if(block::entries_per_block_bits >= codim){
        return gens;
    }
    // printf("a-3\n");
    // todo: deal with weird page sizes
    size_t areasize = sizeof(block) << (codim - block::entries_per_block_bits);
    size_t pagesize = 1 << 12;
    areasize = (areasize + (pagesize - 1)) & ~pagesize;
    block* area = (block*)mmap(0, areasize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    // printf("a-3+1-1\n");
    area[0] = t;
    uint64_t gen = (1 << block::entries_per_block_bits);
    const uint64_t entries_per_thread = blocks_per_slice << block::entries_per_block_bits;
    const uint64_t max_st_entries = std::min((uint64_t)(1ull << codim), entries_per_thread);
    while(gen < max_st_entries){
        uint64_t high_base = (1ull << 63) >> _lzcnt_u64(gen);
        uint64_t high_base_blocks = high_base >> block::entries_per_block_bits;
        uint64_t last_gen = gen;
        // printf("a-3+1\n");
        gen = std::min(slice<block>(dist, gen, high_base_blocks, area, 0, &area[high_base_blocks], high_base), high_base << 1);
        // printf("a-3+2\n");
        gens.push_back(gen);
        // printf("i %05lu v %016lx\n", gens.size() - 1, gen);
        fflush(stdout);
        if(last_gen >= gen){
            printf("sussy 2 %08lx %08lx\n", last_gen, gen);
            return gens;
        }
    }
    // printf("a-2\n");
    // while(gen < (1ull << codim)){
    //     // uint64_t high_base = (1ull << 63) >> _lzcnt_u64(gen);
    //     // uint64_t high_base_blocks = high_base >> block::entries_per_block_bits;
    //     // uint64_t base_mask = - (1ull << entries_per_thread);
    //     uint64_t last_gen = gen;
    //     // gen = std::min(slice<block>(dist, gen, high_base_blocks, area, 0, &area[high_base_blocks], high_base), high_base << 1);
    //     std::atomic_uint64_t next_slice = {0};
    //     std::atomic_uint64_t next_gen = {(1ull << 63) >> (_lzcnt_u64(gen) - 1)};
    //     // printf("a-1\n");
    //     std::thread threads[max_threads];
    //     // printf("a\n");
    //     std::chrono::time_point<std::chrono::steady_clock> start, end;
    //     if((gen & (gen - 1)) == 0){
    //         start = std::chrono::steady_clock::now();
    //     }
    //     // uint64_t thread_count = std::min(max_threads, )
    //     for(uint32_t i = 0; i < max_threads; i++){
    //         threads[i] = std::thread(findcode_thread<block>, dist, gen, area, blocks_per_thread, &next_slice, &next_gen);
    //         // printf("b %d\n", i);
    //     }
    //     for(uint32_t i = 0; i < max_threads; i++){
    //         threads[i].join();
    //     }
    //     if((gen & (gen - 1)) == 0){
    //         end = std::chrono::steady_clock::now();
    //         uint64_t check_bits = 64 - _lzcnt_u64(gen);
    //         printf("%2.3f ms for %lu check bits\n", ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (1000), check_bits);
    //     }
    //     gen = next_gen.load(std::memory_order_relaxed);
    //     gens.push_back(gen);
    //     // printf("i %05lu v %016lx\n", gens.size() - 1, gen);
    //     fflush(stdout);
    //     if(last_gen >= gen){
    //         printf("sussy 2 %08lx %08lx\n", last_gen, gen);
    //         return gens;
    //     }
    // }
    std::atomic_uint64_t next_slice = {0};
    std::atomic_uint64_t next_gen = {(1ull << 63) >> (_lzcnt_u64(gen) - 1)};
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    start = std::chrono::steady_clock::now();
    uint64_t last_gen = 0;
    auto update_gen = [&start, &end, &last_gen, &gen, &gens, &next_slice, &next_gen](){
        last_gen = gen;
        gen = next_gen.load();
        if((gen & (gen - 1)) == 0){
            start = std::chrono::steady_clock::now();
        }
        if((last_gen & (last_gen - 1)) == 0){
            end = std::chrono::steady_clock::now();
            uint64_t check_bits = 64 - _lzcnt_u64(gen);
            printf("%2.3f ms for %lu check bits\n", ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (1000), check_bits);
        }
        gens.push_back(gen);
        next_gen.store((1ull << 63) >> (_lzcnt_u64(gen) - 1));
        next_slice.store(0);
    };
    std::barrier<decltype(update_gen)> sync_point(max_threads, update_gen);
    auto work = [dist, codim, &gen, &sync_point, area, blocks_per_slice, &next_slice, &next_gen](){
        while(gen < (1ull << codim)){
            uint64_t local_gen = gen;
            uint64_t high_base_entry = (1ull << 63) >> _lzcnt_u64(local_gen);
            uint64_t high_base_block = high_base_entry >> block::entries_per_block_bits;
            uint64_t slices = 0;
            uint64_t local_next_gen = std::numeric_limits<uint64_t>::max();
            for(;;){
                uint64_t this_slice = next_slice.fetch_add(1, std::memory_order_relaxed);
                uint64_t slice_first_block = this_slice * blocks_per_slice;
                uint64_t slice_first_entry = slice_first_block >> block::entries_per_block_bits;
                if(slice_first_block >= high_base_block){
                    break;
                }
                uint64_t high_block_index = high_base_block + slice_first_block;
                uint64_t low_block_index = ((local_gen >> block::entries_per_block_bits) ^ high_block_index) & (-blocks_per_slice);
                local_next_gen = std::min(local_next_gen, slice<block>(dist, local_gen, blocks_per_slice, &area[low_block_index], low_block_index << block::entries_per_block_bits, &area[high_block_index], high_block_index << block::entries_per_block_bits));
                slices++;
            }
            // atomic_min plz
            uint64_t cur_next_gen = next_gen.load(std::memory_order_relaxed);
            while(cur_next_gen > local_next_gen){
                next_gen.compare_exchange_weak(cur_next_gen, local_next_gen, std::memory_order_relaxed, std::memory_order_relaxed);
            }
            sync_point.arrive_and_wait();
        }
    };
    std::thread threads[max_threads];
    for(uint32_t i = 0; i < max_threads; i++){
        // template<typename block, class blah> void findcode_thread2(uint32_t dist, uint32_t codim, uint64_t& gen,
        // std::barrier<blah> sync_point, block* area, uint64_t blocks_per_slice, std::atomic_uint64_t next_slice, std::atomic_uint64_t next_gen){
        // template<typename block, class blah> void findcode_thread2(uint32_t dist, uint32_t codim, uint64_t& gen,
        // std::barrier<blah> sync_point, block* area, uint64_t blocks_per_slice, std::atomic_uint64_t next_slice, std::atomic_uint64_t next_gen){
        // template<typename block, class blah> void findcode_thread2(uint32_t dist, uint32_t codim, uint64_t* gen,
        // std::barrier<blah> sync_point, block* area, uint64_t blocks_per_slice, std::atomic_uint64_t* next_slice, std::atomic_uint64_t* next_gen){
        // threads[i] = std::thread(findcode_thread2<block, decltype(update_gen)>, dist, codim, &gen, sync_point, area, blocks_per_slice, &next_slice, &next_gen);
        threads[i] = std::thread(work);
    }
    for(uint32_t i = 0; i < max_threads; i++){
        threads[i].join();
    }
    munmap((void*)area, areasize);
    return gens;
}

int main(){
    // x4a_decompress(0x38);
    // for(int i = 0; i < 256; i++){
    //     if(!eq(x4a_decompress(i), x4a_decompress(x4a_compress(x4a_decompress(i))))){
    //         printf("%x %x\n", i, x4a_compress(x4a_decompress(i)));
    //         print(x4a_decompress(i));
    //         print(x4a_decompress(x4a_compress(x4a_decompress(i))));
    //         return 1;
    //     }
    // }
    // printf("hi\n");
    // std::vector<uint64_t> gens = findcode2<x4a_skx_zmm_block>(5, 25);
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    uint64_t thread_counts[] = {8, 192};
            for(int i = 30; i < 31; i++){
    for(int j = 0; j < 2; j++){
        printf("%lu threads:\n", thread_counts[j]);
        for(int k = 8; k < 16; k++){
            printf("%u blocks per slice\n", 1 << k);
                start = std::chrono::steady_clock::now();
                std::vector<uint64_t> gens = findcode2<x4a_icx_zmm_block>(5, i, 1 << k, thread_counts[j]);
                end = std::chrono::steady_clock::now();
                printf("Generators for codim %u: %zu\n", i, gens.size());
                printf("%2.3f ms\n", ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (1000));
            }
        }
    }
    // d5.txt skips the first gen
    // for(int i = 1; i < gens.size(); i++){
    //     printf("i %05zu v %016lx\n", i, gens[i]);
    // }
    // bake(x4a_icx_zmm_block::gen_max_table(0));
    // bake(x4a_icx_zmm_block::gen_max_table(1));
    // bake(x4a_skx_ymm_block::gen_zpos_table());
    // __m512i x = mm512_iota_epi8();
    // __m512i y = _mm512_set1_epi64(0x8040201008040201);
    // __m512i z = mm512_gf2p8affine_epi64_epi8(x, y, 0);
    // print(z);
}