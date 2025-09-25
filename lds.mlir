// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @lds_load_kernel(%in_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %offs_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %out_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<32x1xi32, #blocked>
    %offs_m = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %offs_0 = arith.muli %offs, %cst : tensor<32x1xi32, #blocked>
    %offs_1 = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %offs_2 = tt.broadcast %offs_0 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %offs_3 = tt.broadcast %offs_1 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %offs_4 = arith.addi %offs_2, %offs_3 : tensor<32x64xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable>
    %0 = amdgpu.buffer_load_to_local %in_ptr[%offs_4] into %smem : <bf16>[tensor<32x64xi32, #blocked>]  -> <32x64xbf16, #shared, #smem, mutable>
    %offsets = amdgpu.buffer_load %offs_ptr[%offs_4] : tensor<32x64xi32, #blocked>
    %a = "amdgpu.lds_load"(%smem, %offsets) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (!ttg.memdesc<32x64xbf16, #shared, #smem, mutable>, tensor<32x64xi32, #blocked>) -> tensor<32x64xbf16, #blocked>
    amdgpu.buffer_store %a, %out_ptr[%offs_4] : tensor<32x64xbf16, #blocked>
    tt.return
  }
}
