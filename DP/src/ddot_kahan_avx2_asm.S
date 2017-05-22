# mark_description "Intel(R) C Intel(R) 64 Compiler XE for applications running on Intel(R) 64, Version 13.1.3.192 Build 2013060";
# mark_description "7";
# mark_description "-I./INTEL/ -I./includes -c -D_GNU_SOURCE -DTIMINGLEV=1 -Ofast -std=c99 -xHost -fno-alias -openmp -S -fsource";
# mark_description "-asm -masm=intel -o INTEL/ddot_kahan_avx2_intrin.s";
	.intel_syntax noprefix
	.file "ddot_kahan_avx2_intrin.c"
	.text
..TXTST0:
# -- Begin  ddot_kahan_avx2_intrin
# mark_begin;
       .align    16,0x90
	.globl ddot_kahan_avx2_asm
ddot_kahan_avx2_asm:
# parameter 1: edi
# parameter 2: rsi
# parameter 3: rdx
# parameter 4: rcx
..B1.1:                         # Preds ..B1.0

### {

..___tag_value_ddot_kahan_avx2_intrin.1:                        #12.1
        push      rbp                                           #12.1
..___tag_value_ddot_kahan_avx2_intrin.3:                        #
        mov       rbp, rsp                                      #12.1
..___tag_value_ddot_kahan_avx2_intrin.4:                        #
        and       rsp, -32                                      #12.1
        push      rbx                                           #12.1
        sub       rsp, 184                                      #12.1
..___tag_value_ddot_kahan_avx2_intrin.6:                        #
        mov       rbx, rdx                                      #12.1

### 
###     if (N == 0)

        test      edi, edi                                      #14.14
        je        ..B1.14       # Prob 28%                      #14.14
                                # LOE rcx rbx rsi r12 r13 r14 r15 edi
..B1.2:                         # Preds ..B1.1

###         return 0.0f;
### 
###     __m256d sum1, sum2, sum3, sum4, sum5;
###     sum1 = _mm256_set1_pd(10.0);
###     sum2 = _mm256_set1_pd(20.0);
###     sum3 = _mm256_set1_pd(40.0);
###     sum4 = _mm256_set1_pd(80.0);
###     sum5 = _mm256_set1_pd(160.0);
###     __m256d c1, c2, c3, c4, c5;
###     c1 = _mm256_set1_pd(320.0);
###     c2 = _mm256_set1_pd(640.0);
###     c3 = _mm256_set1_pd(1280.0);
###     c4 = _mm256_set1_pd(2560.0);
###     c5 = _mm256_set1_pd(5120.0);
###     __m256d one;
###     one = _mm256_set1_pd(1.0);
### 
###     int i, rem;
###     rem = N % 20;

        mov       eax, 1717986919                               #33.15
        mov       r9d, edi                                      #33.15
        imul      edi                                           #33.15
        sar       r9d, 31                                       #33.15

### 
###     __m256d prod1, y1, t1, a1, b1;
###     __m256d prod2, y2, t2, a2, b2;
###     __m256d prod3, y3, t3, a3, b3;
###     __m256d prod4, y4, t4, a4, b4;
###     __m256d prod5, y5, t5, a5, b5;
### 
###     /* use four way unrolling */
###     for (i=0; i<N-rem; i+=20) {

        mov       r8d, edi                                      #42.17
        sar       edx, 3                                        #33.15
        sub       edx, r9d                                      #33.15
        xor       r9d, r9d                                      #42.10
        vxorpd    ymm6, ymm6, ymm6
        vmovapd   ymm5, ymm6
        vmovapd   ymm10,ymm6 
        vmovapd   ymm9, ymm6 
        vmovapd   ymm7, ymm6 
        vmovapd   ymm1, ymm6 
        vmovapd   ymm2, ymm6 
        vmovapd   ymm3, ymm6 
        vmovapd   ymm4, ymm6 
        vmovapd   ymm8, ymm6 
        vmovapd   ymm0, YMMWORD PTR .L_2il0floatpacket.14[rip]  #30.11
        lea       eax, DWORD PTR [rdx+rdx*4]                    #33.15
        shl       eax, 2                                        #33.15
        neg       eax                                           #33.15
        add       eax, edi                                      #33.15
        sub       r8d, eax                                      #42.17
        test      r8d, r8d                                      #42.19
        jle       ..B1.6        # Prob 16%                      #42.19
                                # LOE rcx rbx rsi r9 r12 r13 r14 r15 eax edi ymm0 ymm1 ymm2 ymm3 ymm4 ymm5 ymm6 ymm7 ymm8 ymm9 ymm10
..B1.3:                         # Preds ..B1.2
        movsxd    r8, edi                                       #42.5
        movsxd    rax, eax                                      #42.5
        sub       r8, rax                                       #42.17
                                # LOE rcx rbx rsi r8 r9 r12 r13 r14 r15 eax edi ymm0 ymm1 ymm2 ymm3 ymm4 ymm5 ymm6 ymm7 ymm8 ymm9 ymm10
..B1.4:                         # Preds ..B1.4 ..B1.3

###         /* load 4x4 doubles into four vector registers */
###         a1 = _mm256_load_pd(&a[i]);
###         a2 = _mm256_load_pd(&a[i+4]);
###         a3 = _mm256_load_pd(&a[i+8]);

        #define sum1    ymm6
        #define sum2    ymm5
        #define sum3    ymm10
        #define sum4    ymm9
        #define sum5    ymm7

        #define c1      ymm1
        #define c2      ymm2
        #define c3      ymm3
        #define c4      ymm4
        #define c5      ymm8

        #define one     ymm0

        #define tmp1    ymm11
        #define tmp2    ymm12
        #define tmp3    ymm13
        #define tmp4    ymm14
        #define tmp5    ymm15

        ##         double prod = a[i]*b[i];
        ##         double y = prod-c;
        vmovapd  tmp1, [rsi+r9*8]                   # tmp1 = a[i]
        vmovapd  tmp2, [rsi+r9*8+32]                # tmp2 = a[i+4]
        vmovapd  tmp3, [rsi+r9*8+64]                # tmp3 = a[i+8]
        vmovapd  tmp4, [rsi+r9*8+96]                # tmp4 = a[i+12]
        vmovapd  tmp5, [rsi+r9*8+128]               # tmp5 = a[i+16]
        #            c(y) = tmp1(a)*[rbx+r9*8+xx](b)-c
        vfmsub231pd     c1, tmp1, [rbx+r9*8]        # c1/y = tmp1/a[i] * [rbx+r9*8]/b[i] - c1/c1
        vfmsub231pd     c2, tmp2, [rbx+r9*8+32]     # c2/y = tmp2/a[i+4] * [rbx+r9*8]/b[i+4] - c2/c2
        vfmsub231pd     c3, tmp3, [rbx+r9*8+64]     # ...
        vfmsub231pd     c4, tmp4, [rbx+r9*8+96]     # ...
        vfmsub231pd     c5, tmp5, [rbx+r9*8+128]    # ...

        ##        double t = sum+y;
        ##        N.B. can't use fmadd, because one of the three operands is
        ##        destination register, but we still need sum* and y*
        ##        N.B. c holds y!
        vaddpd     tmp1, sum1, c1                   # tmp1/t1 = sum1 + c1/y1
        vaddpd     tmp2, sum2, c2                   # tmp2/t2 = sum2 + c2/y2
        vaddpd     tmp3, sum3, c3                   # ...
        vaddpd     tmp4, sum4, c4                   # ...
        vaddpd     tmp5, sum5, c5                   # ...

        ##        c = (t-sum)-y;
        ##          becomes
        ##          tmp = t - sum
        ##          c = tmp - y
        ##            sum(tmp) = tmp(t) * 1.0 - sum
        vfmsub231pd      sum1, tmp1, one            # sum1/tmp1 = tmp1/t1 * 1.0f - sum1
        vfmsub231pd      sum2, tmp2, one            # sum2/tmp2 = tmp2/t2 * 1.0f - sum2
        vfmsub231pd      sum3, tmp3, one            # ...
        vfmsub231pd      sum4, tmp4, one            # ...
        vfmsub231pd      sum5, tmp5, one            # ...
        ##          N.B. c holds y!
        vsubpd          c1, sum1, c1                # c1/c1 = sum1/tmp1 - c1/y1
        vsubpd          c2, sum2, c2                # c2/c2 = sum2/tmp2 - c1/y2
        vsubpd          c3, sum3, c3                # ...
        vsubpd          c4, sum4, c4                # ...
        vsubpd          c5, sum5, c5                # ...
        vmovapd         sum1, tmp1                  # sum1/t1 = tmp1/t1
        vmovapd         sum2, tmp2                  # sum2/t2 = tmp2/t2
        vmovapd         sum3, tmp3                  #
        vmovapd         sum4, tmp4                  #
        vmovapd         sum5, tmp5                  #
        add       r9, 20                                        #42.24
        cmp       r9, r8                                        #42.19
        jl        ..B1.4        # Prob 82%                      #42.19
                                # LOE rcx rbx rsi r8 r9 r12 r13 r14 r15 eax edi ymm0 ymm1 ymm2 ymm3 ymm4 ymm5 ymm6 ymm7 ymm8 ymm9 ymm10
..B1.6:                         # Preds ..B1.4 ..B1.2

###     }
### 
###     /* reduce five simd vectors to one simd vector using Kahan */
###     c1 = _mm256_sub_pd(c1, c2);

        vsubpd    ymm1, ymm1, ymm2                              #84.10

###     c3 = _mm256_sub_pd(c3, c4);
### 
###     y1 = _mm256_sub_pd(sum2, c1);
###     y3 = _mm256_sub_pd(sum4, c3);
###     t1 = _mm256_add_pd(sum1, y1);
###     t3 = _mm256_add_pd(sum3, y3);
###     c1 = _mm256_sub_pd(_mm256_sub_pd(t1, sum1), y1);
###     c3 = _mm256_sub_pd(_mm256_sub_pd(t3, sum3), y3);
###     sum1 = t1;
###     sum3 = t3;
### 
###     c1 = _mm256_sub_pd(c1, c3);
###     y1 = _mm256_sub_pd(sum3, c1);
###     t1 = _mm256_add_pd(sum1, y1);
###     c1 = _mm256_sub_pd(_mm256_sub_pd(t1, sum1), y1);
###     sum1 = t1;
### 
###     c1 = _mm256_sub_pd(c1, c5);
###     y1 = _mm256_sub_pd(sum5, c1);
###     t1 = _mm256_add_pd(sum1, y1);
###     c1 = _mm256_sub_pd(_mm256_sub_pd(t1, sum1), y1);
###     sum1 = t1;
### 
###     /* store results of vector register onto stack */
###     double tmp[4];
###     double c_tmp[4];
###     _mm256_store_pd(&tmp[0], sum1);
###     _mm256_store_pd(&c_tmp[0], c1);
### 
###     double sum = 0.0;
###     double c = c_tmp[0] + c_tmp[1] + c_tmp[2] + c_tmp[3];
### 
###     /* perform scalar Kahan sum of partial sums */
### #pragma novector
###     for (i=0; i<4; ++i) {
###         double y = tmp[i]-c;
###         double t = sum+y;
###         c = (t-sum)-y;
###         sum = t;
###     }
### 
###     /* perform scalar Kahan sum of loop remainer */
### #pragma novector
###     for (i=N-rem; i<N; ++i) {

        mov       r8d, eax                                      #128.5
        vsubpd    ymm0, ymm3, ymm4                              #85.10
        vsubpd    ymm2, ymm5, ymm1                              #87.10
        vsubpd    ymm3, ymm9, ymm0                              #88.10
        vaddpd    ymm12, ymm6, ymm2                             #89.10
        vaddpd    ymm9, ymm10, ymm3                             #90.10
        vsubpd    ymm6, ymm12, ymm6                             #91.24
        vsubpd    ymm10, ymm9, ymm10                            #92.24
        vsubpd    ymm4, ymm6, ymm2                              #91.10
        vsubpd    ymm5, ymm10, ymm3                             #92.10
        vsubpd    ymm11, ymm4, ymm5                             #96.10
        neg       r8d                                           #128.5
        vsubpd    ymm14, ymm9, ymm11                            #97.10
        vaddpd    ymm1, ymm12, ymm14                            #98.10
        vsubpd    ymm13, ymm1, ymm12                            #99.24
        vsubpd    ymm15, ymm13, ymm14                           #99.10
        vsubpd    ymm8, ymm15, ymm8                             #102.10
        vsubpd    ymm0, ymm7, ymm8                              #103.10
        vaddpd    ymm2, ymm1, ymm0                              #104.10
        vsubpd    ymm7, ymm2, ymm1                              #105.24
        vmovupd   YMMWORD PTR [rsp], ymm2                       #111.22
        vsubpd    ymm1, ymm7, ymm0                              #105.10
        vmovsd    xmm6, QWORD PTR [rsp]                         #120.20
        vmovupd   YMMWORD PTR [32+rsp], ymm1                    #112.22
        vmovsd    xmm10, QWORD PTR [8+rsp]                      #120.20
        vmovsd    xmm11, QWORD PTR [16+rsp]                     #120.20
        vmovsd    xmm1, QWORD PTR [24+rsp]                      #120.20
        vmovsd    xmm3, QWORD PTR [32+rsp]                      #115.16
        vaddsd    xmm4, xmm3, QWORD PTR [40+rsp]                #115.27
        vaddsd    xmm5, xmm4, QWORD PTR [48+rsp]                #115.38
        vaddsd    xmm7, xmm5, QWORD PTR [56+rsp]                #115.49
        vsubsd    xmm8, xmm6, xmm7                              #120.27
        vaddsd    xmm13, xmm8, xmm10                            #121.24
        vsubsd    xmm9, xmm13, xmm8                             #122.16
        vsubsd    xmm12, xmm9, xmm10                            #122.21
        vsubsd    xmm15, xmm11, xmm12                           #120.27
        vaddsd    xmm0, xmm13, xmm15                            #121.24
        vsubsd    xmm14, xmm0, xmm13                            #122.16
        vsubsd    xmm13, xmm14, xmm15                           #122.21
        vsubsd    xmm3, xmm1, xmm13                             #120.27
        vaddsd    xmm1, xmm0, xmm3                              #121.24
        vsubsd    xmm2, xmm1, xmm0                              #122.16
        vsubsd    xmm0, xmm2, xmm3                              #122.21
        jns       ..B1.13       # Prob 50%                      #128.21
                                # LOE rcx rbx rsi r12 r13 r14 r15 eax edi xmm0 xmm1
..B1.7:                         # Preds ..B1.6
        mov       r9d, eax                                      #128.5
        mov       r10d, 1                                       #128.5
        shr       r9d, 31                                       #128.5
        xor       r8d, r8d                                      #128.5
        add       r9d, eax                                      #128.5
        sar       r9d, 1                                        #128.5
        test      r9d, r9d                                      #128.5
        jbe       ..B1.11       # Prob 10%                      #128.5
                                # LOE rcx rbx rsi r12 r13 r14 r15 eax edi r8d r9d r10d xmm0 xmm1
..B1.8:                         # Preds ..B1.7

###         double prod = a[i]*b[i];

        movsxd    rdi, edi                                      #129.23
        movsxd    rdx, eax                                      #129.23
        shl       rdx, 3                                        #129.28
        lea       r11, QWORD PTR [rbx+rdi*8]                    #129.28
        sub       r11, rdx                                      #129.28
        lea       r10, QWORD PTR [rsi+rdi*8]                    #129.23
        sub       r10, rdx                                      #129.23
        .align    16,0x90
                                # LOE rcx rbx rsi r10 r11 r12 r13 r14 r15 eax edi r8d r9d xmm0 xmm1
..B1.9:                         # Preds ..B1.9 ..B1.8

###         double y = prod-c;

        lea       edx, DWORD PTR [r8+r8]                        #130.25
        inc       r8d                                           #128.5
        movsxd    rdx, edx                                      #129.28
        vmovsd    xmm2, QWORD PTR [r10+rdx*8]                   #129.23
        vmulsd    xmm3, xmm2, QWORD PTR [r11+rdx*8]             #129.28
        vmovsd    xmm4, QWORD PTR [8+r10+rdx*8]                 #129.23
        vsubsd    xmm0, xmm3, xmm0                              #130.25
        vmulsd    xmm5, xmm4, QWORD PTR [8+r11+rdx*8]           #129.28

###         double t = sum+y;

        vaddsd    xmm7, xmm1, xmm0                              #131.24

###         c = (t-sum)-y;

        vsubsd    xmm1, xmm7, xmm1                              #132.16
        vsubsd    xmm6, xmm1, xmm0                              #132.21
        vsubsd    xmm9, xmm5, xmm6                              #130.25
        vaddsd    xmm1, xmm9, xmm7                              #131.24
        vsubsd    xmm8, xmm1, xmm7                              #132.16
        vsubsd    xmm0, xmm8, xmm9                              #132.21
        cmp       r8d, r9d                                      #128.5
        jb        ..B1.9        # Prob 64%                      #128.5
                                # LOE rcx rbx rsi r10 r11 r12 r13 r14 r15 eax edi r8d r9d xmm0 xmm1
..B1.10:                        # Preds ..B1.9
        lea       r10d, DWORD PTR [1+r8+r8]                     #128.5
                                # LOE rcx rbx rsi r12 r13 r14 r15 eax edi r10d xmm0 xmm1
..B1.11:                        # Preds ..B1.10 ..B1.7
        lea       r8d, DWORD PTR [-1+r10]                       #128.5
        cmp       eax, r8d                                      #128.5
        jbe       ..B1.13       # Prob 10%                      #128.5
                                # LOE rcx rbx rsi r12 r13 r14 r15 eax edi r10d xmm0 xmm1
..B1.12:                        # Preds ..B1.11
        movsxd    rdi, edi                                      #129.23
        movsxd    rax, eax                                      #129.23
        movsxd    r10, r10d                                     #129.23
        sub       rdi, rax                                      #129.28
        add       rdi, r10                                      #130.25
        vmovsd    xmm2, QWORD PTR [-8+rsi+rdi*8]                #129.23
        vmulsd    xmm3, xmm2, QWORD PTR [-8+rbx+rdi*8]          #129.28
        vsubsd    xmm0, xmm3, xmm0                              #130.25
        vaddsd    xmm4, xmm1, xmm0                              #131.24
        vsubsd    xmm1, xmm4, xmm1                              #132.16
        vsubsd    xmm0, xmm1, xmm0                              #132.21

###         sum = t;

        vmovapd   xmm1, xmm4                                    #133.9
                                # LOE rcx r12 r13 r14 r15 xmm0 xmm1
..B1.13:                        # Preds ..B1.12 ..B1.11 ..B1.6

###     }
### 
###     (*r) = sum;

        vmovsd    QWORD PTR [rcx], xmm1                         #136.7

###     return c;

        vzeroupper                                              #137.12
        add       rsp, 184                                      #137.12
..___tag_value_ddot_kahan_avx2_intrin.7:                        #137.12
        pop       rbx                                           #137.12
        mov       rsp, rbp                                      #137.12
        pop       rbp                                           #137.12
..___tag_value_ddot_kahan_avx2_intrin.8:                        #
        ret                                                     #137.12
..___tag_value_ddot_kahan_avx2_intrin.10:                       #
                                # LOE
..B1.14:                        # Preds ..B1.1
        vxorpd    xmm0, xmm0, xmm0                              #15.16
        add       rsp, 184                                      #15.16
..___tag_value_ddot_kahan_avx2_intrin.13:                       #15.16
        pop       rbx                                           #15.16
        mov       rsp, rbp                                      #15.16
        pop       rbp                                           #15.16
..___tag_value_ddot_kahan_avx2_intrin.14:                       #
        ret                                                     #15.16
        .align    16,0x90
..___tag_value_ddot_kahan_avx2_intrin.16:                       #
                                # LOE
# mark_end;
	.type	ddot_kahan_avx2_asm,@function
	.size	ddot_kahan_avx2_asm,.-ddot_kahan_avx2_asm
	.data
# -- End  ddot_kahan_avx2_intrin
	.section .rodata, "a"
	.align 32
	.align 32
.L_2il0floatpacket.4:
	.long	0x00000000,0x40240000,0x00000000,0x40240000,0x00000000,0x40240000,0x00000000,0x40240000
	.type	.L_2il0floatpacket.4,@object
	.size	.L_2il0floatpacket.4,32
	.align 32
.L_2il0floatpacket.5:
	.long	0x00000000,0x40340000,0x00000000,0x40340000,0x00000000,0x40340000,0x00000000,0x40340000
	.type	.L_2il0floatpacket.5,@object
	.size	.L_2il0floatpacket.5,32
	.align 32
.L_2il0floatpacket.6:
	.long	0x00000000,0x40440000,0x00000000,0x40440000,0x00000000,0x40440000,0x00000000,0x40440000
	.type	.L_2il0floatpacket.6,@object
	.size	.L_2il0floatpacket.6,32
	.align 32
.L_2il0floatpacket.7:
	.long	0x00000000,0x40540000,0x00000000,0x40540000,0x00000000,0x40540000,0x00000000,0x40540000
	.type	.L_2il0floatpacket.7,@object
	.size	.L_2il0floatpacket.7,32
	.align 32
.L_2il0floatpacket.8:
	.long	0x00000000,0x40640000,0x00000000,0x40640000,0x00000000,0x40640000,0x00000000,0x40640000
	.type	.L_2il0floatpacket.8,@object
	.size	.L_2il0floatpacket.8,32
	.align 32
.L_2il0floatpacket.9:
	.long	0x00000000,0x40740000,0x00000000,0x40740000,0x00000000,0x40740000,0x00000000,0x40740000
	.type	.L_2il0floatpacket.9,@object
	.size	.L_2il0floatpacket.9,32
	.align 32
.L_2il0floatpacket.10:
	.long	0x00000000,0x40840000,0x00000000,0x40840000,0x00000000,0x40840000,0x00000000,0x40840000
	.type	.L_2il0floatpacket.10,@object
	.size	.L_2il0floatpacket.10,32
	.align 32
.L_2il0floatpacket.11:
	.long	0x00000000,0x40940000,0x00000000,0x40940000,0x00000000,0x40940000,0x00000000,0x40940000
	.type	.L_2il0floatpacket.11,@object
	.size	.L_2il0floatpacket.11,32
	.align 32
.L_2il0floatpacket.12:
	.long	0x00000000,0x40a40000,0x00000000,0x40a40000,0x00000000,0x40a40000,0x00000000,0x40a40000
	.type	.L_2il0floatpacket.12,@object
	.size	.L_2il0floatpacket.12,32
	.align 32
.L_2il0floatpacket.13:
	.long	0x00000000,0x40b40000,0x00000000,0x40b40000,0x00000000,0x40b40000,0x00000000,0x40b40000
	.type	.L_2il0floatpacket.13,@object
	.size	.L_2il0floatpacket.13,32
	.align 32
.L_2il0floatpacket.14:
	.long	0x00000000,0x3ff00000,0x00000000,0x3ff00000,0x00000000,0x3ff00000,0x00000000,0x3ff00000
	.type	.L_2il0floatpacket.14,@object
	.size	.L_2il0floatpacket.14,32
	.data
	.section .note.GNU-stack, ""
// -- Begin DWARF2 SEGMENT .eh_frame
	.section .eh_frame,"a",@progbits
.eh_frame_seg:
	.align 8
	.4byte 0x00000014
	.8byte 0x7801000100000000
	.8byte 0x0000019008070c10
	.4byte 0x00000000
	.4byte 0x00000074
	.4byte 0x0000001c
	.8byte ..___tag_value_ddot_kahan_avx2_intrin.1
	.8byte ..___tag_value_ddot_kahan_avx2_intrin.16-..___tag_value_ddot_kahan_avx2_intrin.1
	.byte 0x04
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.3-..___tag_value_ddot_kahan_avx2_intrin.1
	.2byte 0x100e
	.byte 0x04
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.4-..___tag_value_ddot_kahan_avx2_intrin.3
	.4byte 0x8610060c
	.2byte 0x0402
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.6-..___tag_value_ddot_kahan_avx2_intrin.4
	.8byte 0xffe00d1c380e0310
	.8byte 0xfffffff80d1affff
	.2byte 0x0422
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.7-..___tag_value_ddot_kahan_avx2_intrin.6
	.2byte 0x04c3
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.8-..___tag_value_ddot_kahan_avx2_intrin.7
	.4byte 0xc608070c
	.byte 0x04
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.10-..___tag_value_ddot_kahan_avx2_intrin.8
	.8byte 0x1c380e031010060c
	.8byte 0xf80d1affffffe00d
	.4byte 0x22ffffff
	.2byte 0x0286
	.byte 0x04
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.13-..___tag_value_ddot_kahan_avx2_intrin.10
	.2byte 0x04c3
	.4byte ..___tag_value_ddot_kahan_avx2_intrin.14-..___tag_value_ddot_kahan_avx2_intrin.13
	.4byte 0xc608070c
# End
