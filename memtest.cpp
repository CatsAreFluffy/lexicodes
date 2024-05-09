#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include <x86intrin.h>
void susclear(__m512i* s, int c, size_t n){
    __m512i v = _mm512_set1_epi32(c);
    for(size_t i = 0; i < n >> 6; i++){
        // s[i] = v;
        s[i] = _mm512_set1_epi64(i);
    }
}
int main(){
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    __m512i* x = (__m512i*)mmap(0, 1ull << 32, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
    if(!x){
        printf("mmap is unhappy\n");
    }
    start = std::chrono::steady_clock::now();
    // std::memset((void*)x, 0, 1ull << 32);
    susclear(x, 0, 1ull<<32);
    end = std::chrono::steady_clock::now();
    printf("%2.3f ms\n", ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (1000));
    start = std::chrono::steady_clock::now();
    std::memset((void*)x, 0, 1ull << 32);
    end = std::chrono::steady_clock::now();
    printf("%2.3f ms\n", ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (1000));
}