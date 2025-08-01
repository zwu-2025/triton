#include <vector>

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::gpu {
namespace {

// We use the following nomenclature in this file.
//
//  - ctaLayout: A layout for one block, i.e. input dims [register, lane, warp]
//    for register layouts, and input dims [offset] for shared layouts.
//  - cgaLayout: Arrangement of multiple blocks, i.e. input dims [block].
//
// Note that this is inconsistent with the type name CTALayoutAttr.  That type
// is equivalent to our cgaLayout.
//
// IMO the name CTALayoutAttr is wrong.  If we tried to be consistent anyway,
// then we'd have to rename ctaLayout to "warpLayout".  I think that's more
// confusing than being inconsistent about "cgaLayout", especially when we have
// to consider the size of the warpLayout (surely that's not the "warpSize").

#define S(v) StringAttr::get(ctx, (v))

SmallVector<unsigned> getDefaultMmaOrder(MmaEncodingTrait layout) {
  auto rank = layout.getRepOrderForOperand(0).size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

// TODO Have order be a mandatory argument of standardOutDimNames.
SmallVector<StringAttr> permuteDimNames(const SmallVector<StringAttr> &names,
                                        const SmallVector<unsigned> &order) {
  assert(names.size() == order.size());
  SmallVector<StringAttr> ret;
  for (unsigned i : order) {
    ret.push_back(names[i]);
  }
  return ret;
}

// Make a LinearLayout that maps a block-id to an N-dimensional index.
//
// The tensor is split up into CTAsPerCGA pieces, which are distributed among
// the CTAsPerCGA CTAs (i.e. blocks) in the CGA (i.e. groups).
//
// See the nomenclature note at the top of the file for an explanation of why
// this is called makeCgaLayout when it accepts a CTALayoutAttr.
LinearLayout makeCgaLayout(CTALayoutAttr layout) {
  MLIRContext *ctx = layout.getContext();
  StringAttr kBlock = S("block");

  int rank = layout.getCTAOrder().size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout ret = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    // Start with the most minor dimension, which is order[0].
    int dim = layout.getCTAOrder()[i];
    int split = layout.getCTASplitNum()[dim];
    int ctas = layout.getCTAsPerCGA()[dim];
    assert(ctas % split == 0);
    ret *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
           LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
  }

  // Transpose to standard order (dim0, dim1, ...).
  return ret.transposeOuts(outDimNames);
}

// Combines the layout of a CTA (input dims [register, lane, warp]) with the
// layout of a CGA (i.e. a block), and ensures that the resulting layout has the
// given shape.
//
// See the nomenclature note at the top of the file for why the variable with
// type CTALayoutAttr is called cgaLayoutAttr.
LinearLayout combineCtaCgaWithShape(LinearLayout ctaLayout,
                                    CTALayoutAttr cgaLayoutAttr,
                                    ArrayRef<int64_t> shape) {
  int rank = shape.size();
  assert(ctaLayout.getNumOutDims() == rank);
  assert(cgaLayoutAttr.getCTAOrder().size() == rank);
  MLIRContext *ctx = cgaLayoutAttr.getContext();

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  llvm::SmallDenseMap<StringAttr, int64_t> labeledShape;
  for (auto [dim, size] : llvm::zip(outDimNames, shape)) {
    labeledShape[dim] = size;
  }

  LinearLayout cgaLayout =
      ensureLayoutNotLargerThan(makeCgaLayout(cgaLayoutAttr), labeledShape)
          .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // Calculate the shape of the ctaLayout, which is `shape` divided by the
  // cgaLayout's size.
  llvm::SmallDenseMap<StringAttr, int64_t> ctaShape;
  assert(llvm::to_vector(ctaLayout.getOutDimNames()) ==
         llvm::to_vector(cgaLayout.getOutDimNames()));
  for (auto dim : ctaLayout.getOutDimNames()) {
    ctaShape[dim] =
        std::max(int64_t{1}, labeledShape[dim] / cgaLayout.getOutDimSize(dim));
  }

  ctaLayout = ensureLayoutNotSmallerThan(ctaLayout, ctaShape);
  ctaLayout = ensureLayoutNotLargerThan(ctaLayout, ctaShape);

  LinearLayout ret = (ctaLayout * cgaLayout).transposeOuts(outDimNames);
  for (auto dim : ret.getOutDimNames()) {
    assert(ret.getOutDimSize(dim) == labeledShape[dim]);
  }
  return ret;
}

LinearLayout swizzledSharedToLinearLayout(ArrayRef<int64_t> shape,
                                          SwizzledSharedEncodingAttr shared) {
  MLIRContext *ctx = shared.getContext();

  auto shapePerCTA = getShapePerCTA(shared, shape);

  int rank = shape.size();
  if (rank == 1) {
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shapePerCTA[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }

  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for the 2 most minor dimensions of the layout.  These are
  // the dims that get swizzled.
  assert(shape.size() >= 2);
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shapePerCTA[colDim];
  int numRows = shapePerCTA[rowDim];
  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
  for (int col = 1; col < numCols; col *= 2) {
    bases2D.push_back({0, col});
  }
  for (int row = 1; row < numRows; row *= 2) {
    int vec = shared.getVec();
    int perPhase = shared.getPerPhase();
    int maxPhase = shared.getMaxPhase();
    bases2D.push_back({row, (vec * ((row / perPhase) % maxPhase)) % numCols});
  }
  LinearLayout ctaLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *= LinearLayout::identity1D(shapePerCTA[dim], S("offset"),
                                          outDimNames[dim]);
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}

LinearLayout
sharedToLinearLayoutAMDRotating(ArrayRef<int64_t> shape,
                                AMDRotatingSharedEncodingAttr shared) {
  MLIRContext *ctx = shared.getContext();

  auto shapePerCTA = getShapePerCTA(shared, shape);

  int rank = shape.size();
  if (rank == 1) {
    return combineCtaCgaWithShape(
        LinearLayout::identity1D(shapePerCTA[0], S("offset"), S("dim0")),
        shared.getCTALayout(), shape);
  }

  auto outDimNames = standardOutDimNames(ctx, rank);

  // Construct bases for the 2 most minor dimensions of the layout.  These are
  // the dims that get swizzled.
  assert(shape.size() >= 2);
  int colDim = shared.getOrder()[0];
  int rowDim = shared.getOrder()[1];
  int numCols = shape[colDim];
  int numRows = shape[rowDim];
  StringAttr colDimName = outDimNames[colDim];
  StringAttr rowDimName = outDimNames[rowDim];

  std::vector<std::vector<int>> bases2D;
  for (int col = 1; col < numCols; col *= 2) {
    bases2D.push_back({0, col});
  }
  for (int row = 1; row < numRows; row *= 2) {
    int vec = shared.getVec();
    int perPhase = shared.getPerPhase();
    int maxPhase = shared.getMaxPhase();

    int phase = (row / perPhase) % maxPhase;
    int blockNo = row / maxPhase / perPhase % maxPhase;
    int combinedPhase = phase ^ blockNo;
    bases2D.push_back({row, (vec * combinedPhase) % numCols});
  }
  LinearLayout ctaLayout =
      LinearLayout({{S("offset"), bases2D}}, {rowDimName, colDimName});

  // Add the remaining dimensions.
  for (int i = 2; i < rank; i++) {
    int dim = shared.getOrder()[i];
    ctaLayout *=
        LinearLayout::identity1D(shape[dim], S("offset"), outDimNames[dim]);
  }

  return combineCtaCgaWithShape(ctaLayout, shared.getCTALayout(), shape);
}

// Returns the layout of a single core matrix which tiles the nvmma layout
LinearLayout getCoreMatrixLinearLayout(NVMMASharedEncodingAttr shared,
                                       bool disableSwizzle) {
  auto *ctx = shared.getContext();

  int elemBitWidth = shared.getElementBitWidth();
  int tileWidthBytes = shared.getSwizzlingByteWidth();
  int vec = shared.getVec();
  int perPhase = shared.getPerPhase();
  int maxPhase = shared.getMaxPhase();

  int tileRows = 8;
  int tileCols = 8 * tileWidthBytes / elemBitWidth;
  bool isFp4Padded = shared.getFp4Padded();

  std::vector<std::vector<int>> bases2D;
  for (int col = 1; col < tileCols; col *= 2) {
    if (isFp4Padded) {
      // Each group of 16 offsets consists of 8 "real" and 8 "padded" offsets.
      // We represent the padded layout by mapping 8 padded offsets to the same
      // coordinates as the real ones. When computing the inverse of this LL,
      // the offsets correspoding to the real ones are picked in the image by
      // invertAndCompose.
      int colPacked = col / 16 * 8 + col % 8;
      bases2D.push_back({0, colPacked});
    } else {
      bases2D.push_back({0, col});
    }
  }
  for (int row = 1; row < tileRows; row *= 2) {
    if (disableSwizzle) {
      bases2D.push_back({row, 0});
    } else if (isFp4Padded) {
      int colPadded = vec * ((row / perPhase) % maxPhase);
      int colPacked = colPadded / 16 * 8 + colPadded % 8;
      bases2D.push_back({row, colPacked});
    } else {
      bases2D.push_back({row, vec * ((row / perPhase) % maxPhase)});
    }
  }
  auto outDimNames = standardOutDimNames(ctx, 2);
  return LinearLayout({{S("offset"), bases2D}}, outDimNames);
}

} // namespace

LinearLayout nvmmaSharedToLinearLayout(ArrayRef<int64_t> shape,
                                       NVMMASharedEncodingAttr shared,
                                       bool disableSwizzle) {
  MLIRContext *ctx = shared.getContext();
  int rank = shape.size();
  auto shapePerCTA = getShapePerCTA(shared, shape);
  auto kOffset = S("offset");
  auto tmaShape = triton::nvidia_gpu::getTMABlockShape(shared, shapePerCTA,
                                                       /*packedSize=*/true);
  if (shared.getSwizzlingByteWidth() == 0) {
    auto outDimNames = standardOutDimNames(ctx, rank);
    LinearLayout layout = LinearLayout::identity1D(tmaShape[rank - 1], kOffset,
                                                   outDimNames[rank - 1]);
    for (int i = rank - 2; i >= 0; --i) {
      layout *= LinearLayout::identity1D(tmaShape[i], kOffset, outDimNames[i]);
    }
    layout = ensureLayoutNotSmallerThan(layout, outDimNames, shapePerCTA);
    return combineCtaCgaWithShape(layout, shared.getCTALayout(), shape);
  }
  assert(rank >= 2);

  // Collapse all the outer dim into one. We will then create a layout for this
  // shape and reshape it to the original shape.
  std::array<int64_t, 2> collapsedTmaShape{1, tmaShape.back()};
  for (int i = 0; i + 1 < rank; i++)
    collapsedTmaShape[0] *= tmaShape[i];
  if (shared.getTransposed()) {
    std::swap(collapsedTmaShape[0], collapsedTmaShape[1]);
  }

  auto tileLayout = getCoreMatrixLinearLayout(shared, disableSwizzle);
  auto outDimNames = standardOutDimNames(ctx, 2);
  auto kRow = outDimNames[0];
  auto kCol = outDimNames[1];
  auto tileRows = tileLayout.getOutDimSize(kRow);
  auto tileCols = tileLayout.getOutDimSize(kCol);

  int packingFactor = shared.getFp4Padded() ? 2 : 1;
  if (collapsedTmaShape[1] * packingFactor < tileCols ||
      collapsedTmaShape[0] < tileRows) {
    llvm::errs() << "Illegal shared layout; expected collapsed shapePerCTA to "
                    "be at least ["
                 << tileRows << ", " << (tileCols / packingFactor)
                 << "], collapsedTmaShape: [" << collapsedTmaShape[0] << ", "
                 << collapsedTmaShape[1] << "]\n";
    llvm::report_fatal_error("Illegal shared layout");
  }

  // Distribute the remaining rows and cols.
  auto layout =
      ensureLayoutNotSmallerThan(tileLayout, outDimNames, collapsedTmaShape);

  // Reshape the layout to the N-D pre-transposed shape per CTA.
  SmallVector<int64_t> maybeTransposedTmaShape = tmaShape;
  if (shared.getTransposed()) {
    // Move the outer dim to the inner position.
    // TODO: we should move back to using `order` instead of transposed to make
    // the order more explicit.
    std::rotate(maybeTransposedTmaShape.begin(),
                maybeTransposedTmaShape.begin() + 1,
                maybeTransposedTmaShape.end());
  }
  auto reshapedLayout = reshapeLayout(ctx, layout, maybeTransposedTmaShape);

  if (shared.getTransposed()) {
    SmallVector<int> order = {rank - 1};
    for (int i = 0; i < rank - 1; i++) {
      order.push_back(i);
    }
    reshapedLayout = transposeLinearLayout(reshapedLayout, order);
  }

  reshapedLayout = ensureLayoutNotSmallerThan(
      reshapedLayout, standardOutDimNames(ctx, shapePerCTA.size()),
      shapePerCTA);
  return combineCtaCgaWithShape(reshapedLayout, shared.getCTALayout(), shape);
}

/// Function to generate lane and warp layout for dot operands.
static LinearLayout broadcastedDotOperandLayout(MLIRContext *ctx,
                                                ArrayRef<unsigned> shape,
                                                ArrayRef<unsigned> order,
                                                unsigned kDim,
                                                StringAttr inDimName) {
  // Let warpsPerCTAMma = {2, 2}, then
  // warpsPerCTA = {2, 1} for opA and warpsPerCTA = {1, 2} for opB
  // assume warpOrder = {1, 0}
  // Assume that C is tiled by 2x2 tiles. Since warpOrder={1, 0}, we have that
  // the C is owned as per the following layout:
  // C: 0 | 1
  //    - | -
  //    2 | 3
  // In order to be able to compute C, we need the following warp tiling of
  // A and B:
  // A: 0 1 | 0 1    B: 0 2 | 1 3
  //    - - | - -       - - | - -
  //    2 3 | 2 3       0 2 | 1 3
  // In other words, we need to broadcast along K
  auto rank = shape.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  LinearLayout layout = LinearLayout::empty();

  // We have to broadcast along the inner dimension
  // For A, when moving along M we go from 0 to 2.
  // For B, when moving along N we go from 0 to 1.
  // As such, choosing the order of A {1, 0}, gives us the correct broadcasting
  // Same happens if the warpOrder is {0, 1}, like in Hopper
  for (auto d : order) {
    if (d == kDim) {
      layout *= LinearLayout::zeros1D(shape[d], inDimName, dimNames[d]);
    } else {
      layout *= LinearLayout::identity1D(shape[d], inDimName, dimNames[d]);
    }
  }
  return layout;
}

LinearLayout
AMDMfmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  int rank = shape.size();
  assert(rank == getRank());

  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;
  int nIndex = 1 + hasBatchDim;
  (void)mIndex, (void)nIndex;

  MLIRContext *ctx = getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  // https://github.com/ROCm/amd_matrix_instruction_calculator can print the
  // register and lane layout for mfma instructions.

  // We use the order from fastest varying to slowest varying. So each base
  // vector is a tuple of values mapping to matrix C's (N, M[, B]) indices,
  // which will be [1, 0] / [2, 1, 0].
  SmallVector<unsigned> order = getDefaultMmaOrder(*this);
  auto dimM = outDimNames[order[1]];
  auto dimN = outDimNames[order[0]];

  unsigned mDim = getMDim();
  unsigned nDim = getNDim();
  auto elementType = getElementType();
  int height = (elementType && elementType->isF64()) ? 1 : 4;
  constexpr int warpSize = 64;

  bool isTransposed = getIsTransposed();
  // Special case for 64x4 mfma: we always transpose the output to turn
  // the 64x4 mfma into a equalvalent 4x64 mfma and swap operand A and B, so
  // that we can use the mfma broadcast.
  if (mDim == 64 && nDim == 4)
    assert(isTransposed && "64x4 mfma must be transposed");

  int tiles = (mDim * nDim) / (warpSize * height);

  LinearLayout tileLayout = LinearLayout::empty();
  if (!isTransposed) {
    // Each lane holds 'height' elements along the M dimension.
    LinearLayout regs = LinearLayout::identity1D(height, kRegister, dimM);
    // First, distribute the lanes along the N dimension.
    // Then, distribute the lanes along the M dimension. If the #elements
    // exceeds the mDim, duplicate elements across lanes - this can happen for
    // 4x4 output.
    LinearLayout lanes = LinearLayout::identity1D(nDim, kLane, dimN) *
                         LinearLayout::identity1D(warpSize / nDim, kLane, dimM);
    tileLayout = (regs * lanes);

    // Repeat the above distribution along the M dimension to fits the tile.
    if (tiles > 0)
      tileLayout *= LinearLayout::identity1D(tiles, kRegister, dimM);
  } else {
    // For the transposed output, we will use the same method for layout but
    // swap the order of the M and N dimensions.
    LinearLayout regs = LinearLayout::identity1D(height, kRegister, dimN);
    LinearLayout lanes = LinearLayout::identity1D(mDim, kLane, dimM) *
                         LinearLayout::identity1D(warpSize / mDim, kLane, dimN);
    tileLayout = (regs * lanes);

    if (tiles > 0)
      tileLayout *= LinearLayout::identity1D(tiles, kRegister, dimN);
  }

  tileLayout = tileLayout.transposeOuts({dimN, dimM});

  // Instead of defining the layout on a CTA tile and using the
  // combineCtaCgaWithShape function to extend it to the whole tensor, we take a
  // different approach. Suppose tilesPerWarp is 2x2—meaning a warp computes a
  // 2x2 block of MFMA tiles. If we define the layout only on the CTA tile and
  // extend it across the tensor, the resulting tile order won’t be N-contiguous
  // (i.e., row-major). Due to the 2x2 shape, the third tile would fall in the M
  // dimension. While defining the layout per CTA tile might seem more
  // intuitive, the current dot op lowering assumes an N-contiguous ordering of
  // MFMA tiles across the entire tensor. In other words, the lowering logic
  // isn't layout-aware, it only supports a fixed N-contiguous MFMA tile
  // ordering. Supporting other orderings would require extending the dot
  // lowering implementation. For now, we conform to the current lowering
  // algorithm by defining the MFMA linear layout globally, with N-contiguous
  // tiles across the tensor and across CTA tile boundaries.
  auto tilesPerWarp = getTilesPerWarp();
  auto warpsPerCTA = getWarpsPerCTA();

  const unsigned tilesPerWarpM = tilesPerWarp[mIndex];
  const unsigned tilesPerWarpN = tilesPerWarp[nIndex];
  const unsigned warpsPerCTAM = warpsPerCTA[mIndex];
  const unsigned warpsPerCTAN = warpsPerCTA[nIndex];

  // First, extend the layout along the N dimension:
  // - registers are distributed across tilesPerWarpN
  // - then across warpsPerCTAN in the N dimension.
  tileLayout *= LinearLayout::identity1D(tilesPerWarpN, kRegister, dimN);
  tileLayout *= LinearLayout::identity1D(warpsPerCTAN, kWarp, dimN);

  // At this point, the layout is defined across the N dimension within a CTA
  // tile. Instead of switching to the M dimension now, we continue extending
  // the layout along the remaining N dimension, and only then proceed along M,
  // following the tilesPerWarp configuration.
  // If the N dimension is not large enough to span multiple CTA tiles (i.e.,
  // the first argument is 0), an empty layout is created, so this identity
  // layout will not introduce any new registers.
  tileLayout *= LinearLayout::identity1D(
      shape[nIndex] / (getNDim() * warpsPerCTAN * tilesPerWarpN), kRegister,
      dimN);
  tileLayout *= LinearLayout::identity1D(tilesPerWarpM, kRegister, dimM);

  // Finally, extend the layout across warps in the M dimension.
  // After this step, the layout covers a sub-tensor of size ctaTileM × N,
  // i.e., the full N dimension and a CTA tile's extent in M.
  // The rest of the layout will be defined by combineCtaCgaWithShape.
  tileLayout *= LinearLayout::identity1D(warpsPerCTAM, kWarp, dimM);

  // Adjust spatial ordering if batch dimension is present
  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
    tileLayout *=
        LinearLayout::identity1D(warpsPerCTA[0], kWarp, outDimNames[order[2]]);
  }

  return combineCtaCgaWithShape(tileLayout, getCTALayout(), shape);
}

LinearLayout chooseDotDsReadB64TrLayout(DotOperandEncodingAttr dotMfmaLayout,
                                        ArrayRef<int64_t> shape,
                                        int32_t elemBitWidth) {
  auto mfmaLayout = llvm::cast<AMDMfmaEncodingAttr>(dotMfmaLayout.getParent());
  auto mDim = mfmaLayout.getMDim();
  assert(mDim == 16 || mDim == 32);

  bool isFP4 = false;
  if (elemBitWidth == 4) {
    // When doing ds_read_tr4 we actually write the LL as if it were on i8
    // elements this is becasue LL needs to be described for the i8 tensor
    // elements.
    elemBitWidth = 8;
    isFP4 = true;
  }

  assert(elemBitWidth == 16 || elemBitWidth == 8);

  auto rank = shape.size();
  bool hasBatchDim = rank == 3;
  int32_t kWidthDot = dotMfmaLayout.getKWidth();
  auto kDim = dotMfmaLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;

  int32_t kSize = shape[kDim];
  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();

  MLIRContext *ctx = dotMfmaLayout.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  // register order
  // operand A: [1, 0] / [2, 1, 0]
  // operand B: [0, 1] / [1, 2, 0]
  // Regular dot mfma order for both cases is [k, nonk]/[k, nonk, batch]
  // For LDS transpose layout swap order to [nonk, k]/[nonk, k, batch]
  SmallVector<unsigned> order =
      getOrderForDotOperand(dotMfmaLayout.getOpIdx(), rank, /*kContig*/ false);

  std::vector<std::vector<int32_t>> registerBase;
  std::vector<std::vector<int32_t>> laneBase;
  auto populateFP4LL = [&registerBase, &laneBase](int kSize, int mDim) {
    const bool isMfma32 = (mDim == 32);
    // ds_read_b64_tr4 operates on FP4 values swapping the packing of them. Look
    // at i8 values for the ownership of register/lane since it's the data type
    // of the tensor. Register dimension: what i8 in the tile are held by thread
    // 0? Lane dimension: what i8 in the tile are held in register 0 of each
    // thread?
    registerBase.push_back({1, 0});
    registerBase.push_back({2, 0});
    registerBase.push_back({4, 0});
    registerBase.push_back({0, 16});

    // If more than one tile needs to be loaded, populate registerBase
    // dimension for the other tiles
    const int kTileSize = isMfma32 ? 64 : 128;
    for (int reg = kTileSize; reg < kSize; reg *= 2) {
      registerBase.push_back({0, reg});
    }

    // When mDim == 16 we have 16x128 mfma, otherwise it's 16x64
    // The LL for the two is different
    laneBase.push_back({0, 1});
    laneBase.push_back({0, 2});
    laneBase.push_back({0, 4});
    laneBase.push_back({0, 8});
    if (mDim == 16) {
      laneBase.push_back({0, 32});
      laneBase.push_back({0, 64});
    } else {
      assert(mDim == 32);
      laneBase.push_back({8, 0});
      laneBase.push_back({0, 32});
    }
  };
  auto populateLL = [&registerBase, &laneBase](int elemBitWidth, int kSize,
                                               int kWidthDot, int mDim) {
    // Number of bits loaded by an LDS read. ds_read_tr primarily supports
    // 64-bit loads for most element sizes (16b, 8b, 4b).
    const int32_t ldsReadWidth = 64;
    int32_t kWidthTransRead = ldsReadWidth / elemBitWidth;
    const int elemByteWidth = elemBitWidth / 8;
    const bool isMfma32 = (mDim == 32);

    // For ds_read_b64_tr_* instructions, each thread accesses 64 bits (8 bytes)
    // of data. The smallest unit for transposition is a
    // [non-K, K] = {16, kWidthTransRead} sub-tile of elements,
    // where each thread reads kWidthTransRead elements along the non-K
    // dimension. Due to the transposition mechanism, each thread ends up with
    // kWidthTransRead elements along the K dimension.
    //
    // The MFMA selection logic prioritizes double-rate MFMA instructions
    // whenever possible:
    //
    // - For MFMA operations where M = N = 16, when blockK > k, mfma16x16x2*k
    //   is selected; otherwise (blockK ≤ k), mfma16x16xk remains the choice.
    //
    // - For MFMA operations where M = N = 32, when blockK > k, mfma32x32x2*k is
    //   selected; otherwise (blockK ≤ k), mfma32x32xk is used.
    //
    // NOTE: For fp8 and fp4, "double-rate" results in 4*k since scaled MFMA
    // instructions are used.
    //
    // In "double-rate" MFMA instructions, each thread holds 2*kWidthTransRead
    // elements along the K dimension:
    // - The first kWidthTransRead elements belong to the first sub-tile.
    // - The next kWidthTransRead elements belong to the second sub-tile.
    //
    // These elements are then grouped into larger tiles, each consisting of
    // 8 {16, kWidthTransRead} sub-tiles. These tiles correspond to the data
    // for one MFMA instruction. The shape of these tiles depends on the MFMA
    // instruction used.
    //
    // For single-rate MFMA instructions, each thread holds kWidthTransRead
    // elements along the K dimension. This means that the larger tile
    // (corresponding to one MFMA instruction) consists of 4 {16,
    // kWidthTransRead} sub-tiles.

    // Populate register base for first subtile
    for (int i = 1; i < kWidthTransRead; i *= 2) {
      registerBase.push_back({i, 0});
    }

    const int threadsPerSubtileNonK = 16 / kWidthTransRead;
    const int threadsPerSubtileK = kWidthTransRead;

    // Populate lane base for first subtile
    for (int i = 1; i < threadsPerSubtileNonK; i *= 2) {
      laneBase.push_back({i * kWidthTransRead, 0});
    }
    for (int i = 1; i < threadsPerSubtileK; i *= 2) {
      laneBase.push_back({0, i});
    }

    // Function to extend register base for multiple tiles K dim.
    auto extendRegisterBaseForKDim = [&](int kTileSize,
                                         int numSubtilesPerTile) {
      const int regsPerTile = kWidthTransRead * numSubtilesPerTile;
      int totalRegs = (kSize / kTileSize) * regsPerTile;

      for (int reg = regsPerTile; reg < totalRegs; reg *= 2) {
        registerBase.push_back({0, (reg / regsPerTile) * kTileSize});
      }
    };

    // kDoubleTileSize is the k dimension of a tile when double rated
    // mfma instructions are used.
    const int kDoubleTileSize =
        isMfma32 ? 32 / elemByteWidth : 64 / elemByteWidth;
    // kTileSize is the actually k dimention of a tile, which is
    // determined by kWidthDot.
    const int kTileSize = kWidthDot * 64 / mDim;
    // We use kDoubleTileSize as a reference to check whether the given
    // kWidthDot leads to double or single sub-tiles in each tile.
    const int numSubtilesPerTile = (kTileSize == kDoubleTileSize) ? 2 : 1;

    // Extend register base for large K sizes.
    if (numSubtilesPerTile == 2)
      registerBase.push_back({0, threadsPerSubtileK}); // Second subtile

    extendRegisterBaseForKDim(kTileSize, numSubtilesPerTile);

    // Extend lane base based on MFMA size.
    std::vector<std::vector<int32_t>> laneBaseExt;

    if (isMfma32) {
      laneBaseExt = {{16, 0}, {0, numSubtilesPerTile * threadsPerSubtileK}};
    } else {
      laneBaseExt = {{0, numSubtilesPerTile * threadsPerSubtileK},
                     {0, 2 * numSubtilesPerTile * threadsPerSubtileK}};
    }
    laneBase.insert(laneBase.end(), laneBaseExt.begin(), laneBaseExt.end());
  };

  if (isFP4)
    populateFP4LL(kSize, mDim);
  else
    populateLL(elemBitWidth, kSize, kWidthDot, mDim);

  // Base vectors above are defined in a fixed order [non-k-dim, k-dim].
  // To assign them to actual matrix dimensions we associate with register
  // `order` which is also [nonk, k] given we set kContig to false.
  LinearLayout tileLayout({{kRegister, registerBase}, {kLane, laneBase}},
                          {outDimNames[order[0]], outDimNames[order[1]]});
  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  // warp order
  // common for both operand A and B: [0, 1] / [0, 1, 2]
  // in both cases it is [M dim, N dim]/[batch, M dim, N dim]
  auto warpOrder = getDefaultMmaOrder(mfmaLayout);
  LinearLayout warpLayout = identityStandardND(kWarp, warpsPerCTA, warpOrder);

  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);
  return combineCtaCgaWithShape(ctaLayout, mfmaLayout.getCTALayout(), shape);
}

LinearLayout mfmaDotToLinearLayout(DotOperandEncodingAttr dotMfmaLayout,
                                   ArrayRef<int64_t> shape) {
  auto mfmaLayout = llvm::cast<AMDMfmaEncodingAttr>(dotMfmaLayout.getParent());

  auto rank = shape.size();
  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;

  int32_t kWidth = dotMfmaLayout.getKWidth();
  auto kDimIndex = dotMfmaLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;

  auto warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  auto tilesPerWarp = mfmaLayout.getTilesPerWarp();
  auto tilePerWarpNonK = tilesPerWarp[kDimIndex];

  auto mDim = mfmaLayout.getMDim();
  auto nDim = mfmaLayout.getNDim();
  auto opIdx = dotMfmaLayout.getOpIdx();
  auto nonKDim = opIdx == 0 ? mDim : nDim;
  constexpr int warpSize = 64;

  int32_t kSize = shape[kDimIndex];

  MLIRContext *ctx = dotMfmaLayout.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  // register order
  // operand A: [1, 0] / [2, 1, 0]
  // operand B: [0, 1] / [1, 2, 0]
  // for both cases it is [k, nonk]/[k, nonk, batch]
  auto order =
      getOrderForDotOperand(dotMfmaLayout.getOpIdx(), rank, /*kContig*/ true);
  auto dimK = outDimNames[order[0]];
  auto dimNonK = outDimNames[order[1]];

  // warp order
  // common for both operand A and B: [0, 1] / [0, 1, 2]
  // in both cases it is [M dim, N dim]/[batch, M dim, N dim]
  auto warpOrder = getDefaultMmaOrder(mfmaLayout);

  // Each lane holds kWidth elements along the K dimension
  LinearLayout regs = LinearLayout::identity1D(kWidth, kRegister, dimK);
  // First distribute nonKDim elements along the non-K dimension,
  // then distribute remaining elements along the K dimension
  LinearLayout lanes =
      LinearLayout::identity1D(nonKDim, kLane, dimNonK) *
      LinearLayout::identity1D(warpSize / nonKDim, kLane, dimK);
  LinearLayout tileLayout = regs * lanes;

  int kTileSize = warpSize / nonKDim * kWidth;
  // Special case for 4x64 and 64x4 mfma: for the 64x64 operand,
  // we need to repeat the layout 16 times along the K dimension
  if ((mDim == 64 && nDim == 4 && opIdx == 0) ||
      (mDim == 4 && nDim == 64 && opIdx == 1)) {
    tileLayout *= LinearLayout::identity1D(16, kRegister, dimK);
    kTileSize *= 16;
  }

  // If shape K is larger than the tile size, repeat the tile
  // along the K dimension.
  if (kSize > kTileSize) {
    tileLayout *= LinearLayout::identity1D(kSize / kTileSize, kRegister, dimK);
  }

  // Follow the tiles per warp property, repeat the tile layout
  // along the non-K dimension.
  tileLayout *= LinearLayout::identity1D(tilePerWarpNonK, kRegister, dimNonK);

  tileLayout = tileLayout.transposeOuts({dimK, dimNonK});
  if (hasBatchDim) {
    assert(order[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *= LinearLayout::identity1D(1, kRegister, outDimNames[order[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[order[2]]);
  }

  LinearLayout warpLayout = identityStandardND(kWarp, warpsPerCTA, warpOrder);
  LinearLayout ctaLayout = tileLayout * warpLayout;

  // Note the current the output order is [k, nonk]/[k, nonk, batch]. If the
  // layout's out-size is smaller than the shape, we follow this order to
  // extend each dimension to match the shape. After that, we can transpose
  // to match the standard output order.
  return combineCtaCgaWithShape(ctaLayout, mfmaLayout.getCTALayout(), shape)
      .transposeOuts(outDimNames);
}

LinearLayout
AMDWmmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  int rank = shape.size();
  assert(rank == getRank());

  bool hasBatchDim = rank == 3;
  int mIndex = 0 + hasBatchDim;
  int nIndex = 1 + hasBatchDim;
  (void)mIndex, (void)nIndex;

  SmallVector<unsigned> mnkDim = getMNKDimPerInstr();
  unsigned mDim = mnkDim[0], nDim = mnkDim[1];
  (void)mDim, (void)nDim;

  assert(((shape[mIndex] == 1 || shape[mIndex] >= mDim) &&
          (shape[nIndex] == 1 || shape[nIndex] >= nDim)) &&
         "Unsupported tensor shape for given wmma layout");

  MLIRContext *ctx = getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");

  // https://github.com/ROCm/amd_matrix_instruction_calculator can print the
  // register and lane layout for mfma instructions.

  // We use the order from fastest varying to slowest varying. So each base
  // vector is a tuple of values mapping to matrix C's (N, M[, B]) indices.
  auto threadOrder = getMatrixOrder(rank, /*rowMajor*/ !getIsTransposed());
  assert(threadOrder[0] == mIndex || threadOrder[0] == nIndex);
  assert(threadOrder[1] == mIndex || threadOrder[1] == nIndex);

  // For wmma with 16x16 output, each of the 32 threads holds 8 elements.
  //
  // The first version of WMMA layout has following specific:
  // for the register (i.e., element) dimension, these 8 elements are
  // along the matrix C's M dimension, with 1 consecutive elements
  // spanning 1 row and then the next 1 row being a gap.
  //
  // For the lane (i.e., thread) dimension, these threads are along the
  // matrix C's N dimension, with 16 consecutive threads covering a whole
  // row and the next 16 threads start at the next row.
  //
  // The second version of wmma layout is less tricky:
  // for the register dimension 8 elements are along the matrix C's M
  // dimension. First 16 lanes take 0-8 elems along M, second 16 take 8-15.
  // We have 16 pair of threads in each warp, one pair covers the whole
  // column.
  //
  // Please also check explaining comments in TritonGPUAttrDefs.td at the
  // AMDWmmaEncodingAttr section.
  unsigned ver = getVersion();
  assert(ver == 1 || ver == 2);
  LinearLayout tileLayout =
      ver == 1
          ? LinearLayout(
                {{kRegister, {/*gap*/ {0, 2}, {0, 4}, {0, 8}}},
                 {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 1}}}},
                {outDimNames[threadOrder[0]], outDimNames[threadOrder[1]]})
          : LinearLayout(
                {{kRegister, {{0, 1}, {0, 2}, {0, 4}}},
                 {kLane, {{1, 0}, {2, 0}, {4, 0}, {8, 0}, /*gap*/ {0, 8}}}},
                {outDimNames[threadOrder[0]], outDimNames[threadOrder[1]]});

  if (hasBatchDim) {
    int batchIndex = 0;
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *=
        LinearLayout::identity1D(1, kRegister, outDimNames[batchIndex]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[batchIndex]);
  }

  // And each warp takes the same register and lane sub-layout. So multiply with
  // an identity layout for the warp.
  auto warpOrder = getDefaultMmaOrder(*this);
  LinearLayout warpLayout =
      identityStandardND(S("warp"), getWarpsPerCTA(), warpOrder);
  // reorder dim names in rep order, so combineCtaCgaWithShape generate proper
  // extension of layout
  auto repOrder = getRepOrder();
  SmallVector<StringAttr> repDimNames;
  for (auto dim : repOrder)
    repDimNames.push_back(outDimNames[dim]);
  LinearLayout ctaLayout = tileLayout.transposeOuts(repDimNames) *
                           warpLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout wmmaDotOperandToLinearLayout(DotOperandEncodingAttr dotWmmaLayout,
                                          ArrayRef<int64_t> shape) {
  auto wmmaLayout = llvm::cast<AMDWmmaEncodingAttr>(dotWmmaLayout.getParent());
  auto rank = shape.size();
  bool hasBatchDim = rank == 3;
  auto kDim = dotWmmaLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  int32_t kSize = shape[kDim];
  MLIRContext *ctx = dotWmmaLayout.getContext();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  // lane order
  // operand A: [1, 0] / [2, 1, 0]
  // operand B: [0, 1] / [1, 2, 0]
  // for both cases it is [k, nonk]/[k, nonk, batch]
  auto laneOrder =
      getOrderForDotOperand(dotWmmaLayout.getOpIdx(), rank, /*kContig*/ true);
  // generate continuous part of register bases(i.e. kWidth)
  std::vector<std::vector<int32_t>> registerBase;
  const int32_t kWidth = dotWmmaLayout.getKWidth();
  for (int i = 1; i < kWidth; i *= 2)
    registerBase.push_back(std::vector<int32_t>{i, 0});
  std::vector<std::vector<int32_t>> laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}};
  switch (wmmaLayout.getVersion()) {
  case 1:
    // WMMA version 1 duplicates values in lanes 0-15 and 16-31
    laneBase.push_back({0, 0});
    break;
  case 2:
    // WMMA version 2 offset values in lanes 0-15 and 16-31 across k dimensions
    laneBase.push_back({kWidth, 0});
    break;
  default:
    assert(false && "unexpected version");
  }
  // Generate layout for one wmma instruction
  LinearLayout tileLayout(
      {{kRegister, registerBase}, {kLane, laneBase}},
      {outDimNames[laneOrder[0]], outDimNames[laneOrder[1]]});
  if (hasBatchDim) {
    assert(laneOrder[2] == 0);
    // Extend the base vector with one value to accommodate for the batch
    // dimension, which appears at the last.
    tileLayout *=
        LinearLayout::identity1D(1, kRegister, outDimNames[laneOrder[2]]);
    tileLayout *= LinearLayout::identity1D(1, kLane, outDimNames[laneOrder[2]]);
  }

  // Generate warp layout
  auto warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  auto warpOrder = getDefaultMmaOrder(wmmaLayout);
  LinearLayout warpLayout =
      broadcastedDotOperandLayout(ctx, warpsPerCTA, warpOrder, kDim, S("warp"));

  // reorder dim names in rep order, so combineCtaCgaWithShape generate proper
  // extension of layout
  auto repOrder = wmmaLayout.getRepOrderForOperand(dotWmmaLayout.getOpIdx());
  SmallVector<StringAttr> repDimNames;
  for (auto dim : repOrder)
    repDimNames.push_back(outDimNames[dim]);

  // join instruction layout and warps using repetition order of dimensions
  LinearLayout ctaLayout = tileLayout.transposeOuts(repDimNames) *
                           warpLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, wmmaLayout.getCTALayout(), shape);
}

LinearLayout
BlockedEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();
  auto order = getOrder();
  LinearLayout ctaLayout =
      identityStandardND(S("register"), getSizePerThread(), order) *
      identityStandardND(S("lane"), getThreadsPerWarp(), order) *
      identityStandardND(S("warp"), getWarpsPerCTA(), order);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout fmaDotToLinearLayout(DotOperandEncodingAttr operandLayout,
                                  ArrayRef<int64_t> shape) {
  int rank = shape.size();
  auto blocked = cast<BlockedEncodingAttr>(operandLayout.getParent());
  MLIRContext *ctx = operandLayout.getContext();

  // TODO: introduce registerOrder or use getDefaultOrder(operandLayout)
  // Currently this order is used in legacy converter, because we do not
  // have access to full dot operand layout, only parent part.
  auto regOrder = blocked.getOrder();
  auto threadOrder = blocked.getOrder();
  auto warpOrder = blocked.getOrder();
  auto repOrder = blocked.getRepOrder();

  StringAttr kReg = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");

  auto threadSize = llvm::to_vector(blocked.getSizePerThread());
  auto kDimIdx = operandLayout.getOpIdx() == 0 ? rank - 1 : rank - 2;
  threadSize[kDimIdx] = shape[kDimIdx];
  auto threadShape = blocked.getThreadsPerWarp();
  auto warpShape = blocked.getWarpsPerCTA();

  SmallVector<StringAttr> repDimNames =
      permuteDimNames(standardOutDimNames(ctx, rank), repOrder);

  auto registersLayout = identityStandardND(kReg, threadSize, regOrder);
  auto lanesLayout = broadcastedDotOperandLayout(ctx, threadShape, threadOrder,
                                                 kDimIdx, kLane);
  auto warpsLayout =
      broadcastedDotOperandLayout(ctx, warpShape, warpOrder, kDimIdx, kWarp);

  LinearLayout ctaLayout = registersLayout.transposeOuts(repDimNames) *
                           lanesLayout.transposeOuts(repDimNames) *
                           warpsLayout.transposeOuts(repDimNames);

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(operandLayout), shape);
}

LinearLayout nvidiaMmaTile(MLIRContext *ctx, ArrayRef<unsigned> tileShape,
                           unsigned kWidth, ArrayRef<unsigned> order,
                           ArrayRef<unsigned> repOrder) {
  // Trivial layout mapping 0 -> (0, 0), but we set the order to repOrder
  // Like LinearLayout::empty() but with a rank and an order
  int rank = repOrder.size();
  auto dimNames = standardOutDimNames(ctx, rank);
  auto trivialShape = SmallVector<unsigned>(rank, 1);
  LinearLayout ctaLayout =
      identityStandardND(S("register"), trivialShape, repOrder);

  assert(rank >= 2);
  auto inner = order[0];
  auto outer = order[1];

  assert(tileShape.size() == rank);
  int m = tileShape[outer];
  int n = tileShape[inner];

  // The relative order of registers and lanes is given by:
  // - Inner dim: kWidth registers
  // - Inner dim: 4 lanes
  // - Outer dim: 8 lanes
  // - Outer dim: repeat m / 8 times
  // - Inner dim: repeat n / (kWidth * 4) times
  assert(m % 8 == 0);
  assert(n % (kWidth * 4) == 0);
  // There is at least one subtile on the inner-most dimension
  // FIXME. We should implement operator* in terms of operator*=
  // and chain *= instead of using *
  auto outDimNames = llvm::to_vector(ctaLayout.getOutDimNames());
  ctaLayout = ctaLayout *
              LinearLayout::identity1D(kWidth, S("register"), dimNames[inner]) *
              LinearLayout::identity1D(4, S("lane"), dimNames[inner]) *
              LinearLayout::identity1D(8, S("lane"), dimNames[outer]) *
              LinearLayout::identity1D(m / 8, S("register"), dimNames[outer]) *
              LinearLayout::identity1D(n / (kWidth * 4), S("register"),
                                       dimNames[inner]);
  return ctaLayout;
}

LinearLayout
NvidiaMmaEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto ctx = getContext();
  int rank = shape.size();
  assert(rank == getRank());

  SmallVector<unsigned> tileShape;
  if (isAmpere()) {
    // Ampere.getInstrShape() returns the tile shape
    tileShape = SmallVector<unsigned>(getInstrShape());
  } else {
    assert(isHopper());
    auto instrShapeMNK = getInstrShape();
    tileShape = SmallVector<unsigned>({instrShapeMNK[0], instrShapeMNK[1]});
  }
  // nvidiamma layout always assumes kWidth = 2
  constexpr auto kWidth = 2;
  auto order = getDefaultMmaOrder(*this);
  auto ctaLayout = nvidiaMmaTile(ctx, tileShape, kWidth, order, getRepOrder());

  auto warpOrder = getMatrixOrder(rank, /*rowMajor*/ !isHopper());
  ctaLayout *= identityStandardND(S("warp"), getWarpsPerCTA(), warpOrder)
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(), shape);
}

LinearLayout nvidiaDotToLinearLayout(ArrayRef<int64_t> shape,
                                     DotOperandEncodingAttr dot) {
  int rank = shape.size();
  auto mma = cast<NvidiaMmaEncodingAttr>(dot.getParent());
  int kWidth = dot.getKWidth();
  bool isA = dot.getOpIdx() == 0;
  MLIRContext *ctx = mma.getContext();

  SmallVector<unsigned> tileShape(rank, 1);
  if (isA) {
    tileShape[rank - 2] = 16;
    tileShape[rank - 1] = kWidth * 8;
  } else {
    // Hopper takes the rhs via shared memory
    assert(mma.isAmpere());
    tileShape[rank - 2] = kWidth * 8;
    tileShape[rank - 1] = 8;
  }
  auto order = getOrderForDotOperand(dot.getOpIdx(), rank, /*kContig*/ true);
  auto ctaLayout =
      nvidiaMmaTile(ctx, tileShape, kWidth, order, dot.getRepOrder());
  auto kDim = isA ? rank - 1 : rank - 2;
  auto warpOrder = getMatrixOrder(rank, /*rowMajor*/ !mma.isHopper());
  ctaLayout *= broadcastedDotOperandLayout(ctx, mma.getWarpsPerCTA(), warpOrder,
                                           kDim, S("warp"))
                   .transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  return combineCtaCgaWithShape(ctaLayout, getCTALayout(dot), shape);
}

LinearLayout
DotOperandEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  auto parent = getParent();
  if (auto blockedLayout = mlir::dyn_cast<BlockedEncodingAttr>(parent)) {
    return fmaDotToLinearLayout(*this, shape);
  } else if (auto mfmaLayout = mlir::dyn_cast<AMDMfmaEncodingAttr>(parent)) {
    return mfmaDotToLinearLayout(*this, shape);
  } else if (auto wmmaLayout = mlir::dyn_cast<AMDWmmaEncodingAttr>(parent)) {
    return wmmaDotOperandToLinearLayout(*this, shape);
  } else {
    auto mma = mlir::cast<NvidiaMmaEncodingAttr>(parent);
    return nvidiaDotToLinearLayout(shape, *this);
  }
}

LinearLayout SliceEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  MLIRContext *ctx = getContext();

  // First compute the linear layout for this layout's parent.
  SmallVector<int64_t> parentShape(shape);
  parentShape.insert(parentShape.begin() + getDim(), 1);
  LinearLayout parentLL =
      triton::gpu::toLinearLayout(parentShape, getParent(), {});

  // Remove dimension getDim() from the parent layout.
  //
  //  1. Construct a layout `transform` from parent-out-dims to slice-out-dims
  //     that removes the relevant out-dim.
  //  2. Compute linearSlice = parent.compose(transform).  Now linearSlice maps
  //     from parent in-dims to slice out-dims.
  //  3. Fix up duplicate registers introduced by slicing.
  auto outDimNames = standardOutDimNames(ctx, shape.size() + 1);
  LinearLayout transform = LinearLayout::empty();
  for (auto [idx, outDim] : llvm::enumerate(parentLL.getOutDimNames())) {
    if (idx == getDim()) {
      // Because we're multiplying by all zeros, we could replace outDimNames[0]
      // with any other valid out-dim; the layout will be the same.
      transform *= LinearLayout::zeros1D(parentLL.getOutDimSize(outDim), outDim,
                                         outDimNames[0]);
    } else {
      transform *=
          LinearLayout::identity1D(parentLL.getOutDimSize(outDim), outDim,
                                   outDimNames[idx - (idx < getDim() ? 0 : 1)]);
    }
  }
  LinearLayout sliceLL = parentLL.compose(transform);

  // Step 3: Along the "register" dim, remove any all-zero bases.
  auto bases = sliceLL.getBases();
  std::vector<std::vector<int>> newRegBases;
  for (const auto &basis : bases[S("register")]) {
    if (llvm::any_of(basis, [](int b) { return b != 0; })) {
      newRegBases.push_back(basis);
    }
  }
  bases[S("register")] = newRegBases;

  return LinearLayout(std::move(bases),
                      llvm::to_vector(sliceLL.getOutDimNames()));
}

LinearLayout
TritonGPUDialect::toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
                                 ArrayRef<int64_t> allocationShape) {
  CacheKey key{
      std::vector<int64_t>(shape.begin(), shape.end()), layout,
      std::vector<int64_t>(allocationShape.begin(), allocationShape.end())};
  if (auto result = llCache.get(key)) {
    return *result;
  }

  // Layouts are distributed or shared in triton core
  // To add a new layout add an else-if clause
  LinearLayout result = LinearLayout::empty();
  if (auto distributed = dyn_cast<DistributedEncodingTrait>(layout)) {
    assert(allocationShape.empty() &&
           "allocationShape not supported for distributed layout");
    result = distributed.toLinearLayout(shape);
  } else {
    assert(!allocationShape.empty() &&
           "allocationShape not supported for shared layout");
    allocationShape = allocationShape.take_back(shape.size());
    assert(llvm::all_of(allocationShape,
                        [](int64_t dim) {
                          return llvm::isPowerOf2_32(dim) && dim >= 1;
                        }) &&
           "allocationShape must be a postive power of 2");
    assert(llvm::all_of(llvm::zip(allocationShape, shape),
                        [](auto dims) {
                          return std::get<0>(dims) >= std::get<1>(dims);
                        }) &&
           "allocationShape must be at least as large as shape");

    if (auto shared = dyn_cast<SwizzledSharedEncodingAttr>(layout)) {
      result = swizzledSharedToLinearLayout(allocationShape, shared);
    } else if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(layout)) {
      result = nvmmaSharedToLinearLayout(allocationShape, shared);
    } else if (auto sbl = dyn_cast<AMDRotatingSharedEncodingAttr>(layout)) {
      result = sharedToLinearLayoutAMDRotating(allocationShape, sbl);
    } else {
      assert(0 && "unknown layout");
    }
  }

  llCache.set(std::move(key), result);
  return result;
}

LinearLayout toLinearLayout(RankedTensorType type) {
  return toLinearLayout(type.getShape(), type.getEncoding(), {});
}

LinearLayout toLinearLayout(MemDescType type) {
  return toLinearLayout(type.getShape(), type.getEncoding(),
                        type.getAllocShape());
}

LinearLayout toLinearLayout(TensorOrMemDesc type) {
  if (auto ranked = dyn_cast<RankedTensorType>(type)) {
    return toLinearLayout(ranked);
  } else {
    auto memDesc = cast<MemDescType>(type);
    return toLinearLayout(memDesc);
  }
}

LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
                            ArrayRef<int64_t> allocationShape) {
  auto *ctx = layout.getContext();
  return ctx->getLoadedDialect<TritonGPUDialect>()->toLinearLayout(
      shape, layout, allocationShape);
}

LinearLayout getLayoutWithinBlock(const LinearLayout &layout) {
  assert(!layout.getInDimNames().empty());
  MLIRContext *ctx = layout.getInDimNames().begin()->getContext();

  StringAttr kBlock = S("block");
  assert(layout.hasInDim(kBlock));
  auto bases = layout.getBases();
  bases[kBlock] = {};
  return LinearLayout(bases, llvm::to_vector<4>(layout.getOutDimNames()));
}

LinearLayout chooseShemLayoutForRegToRegConversion(
    MLIRContext *ctx, ArrayRef<unsigned> tensorShape,
    ArrayRef<unsigned> repShape, ArrayRef<unsigned> order) {
  auto outDimNames = standardOutDimNames(ctx, tensorShape.size());
  LinearLayout layout = LinearLayout::empty();
  SmallVector<StringAttr> kRepDims;
  SmallVector<StringAttr> kOffsetDims;
  auto totalIters = 1;
  auto totalOffsets = 1;
  for (int i = 0; i < tensorShape.size(); i++) {
    int dim = order[i];
    StringAttr kIteration = S("iteration" + std::to_string(dim));
    StringAttr kOffset = S("offset" + std::to_string(dim));
    kRepDims.push_back(kIteration);
    kOffsetDims.push_back(kOffset);
    assert(llvm::isPowerOf2_32(repShape[dim]));
    assert(llvm::isPowerOf2_32(tensorShape[dim]));
    auto numIters = tensorShape[dim] / repShape[dim];
    layout *=
        LinearLayout::identity1D(repShape[dim], kOffset, outDimNames[dim]);
    layout *= LinearLayout::identity1D(numIters, kIteration, outDimNames[dim]);
    totalIters *= numIters;
    totalOffsets *= repShape[dim];
  }
  StringAttr kOffset = S("offset");
  StringAttr kIteration = S("iteration");
  StringAttr kBlock = S("block");
  SmallVector<StringAttr> newDims;
  newDims.append(kOffsetDims.begin(), kOffsetDims.end());
  newDims.append(kRepDims.begin(), kRepDims.end());
  // Transpose layout from [offset0, rep0, offset1, rep1, ...] to
  // [offset0, offset1, ..., rep0, rep1, ...]
  auto ret = layout.transposeIns(newDims);
  // Reshape layout from [offset0, offset1, ..., rep0, rep1, ...] to
  // [offset, rep, block]
  return ret.reshapeIns(
      {{kOffset, totalOffsets}, {kIteration, totalIters}, {kBlock, 1}});
}

LinearLayout chooseDsReadB64TrLayout(Attribute enc, ArrayRef<int64_t> shape,
                                     int32_t elemBitWidth) {
  auto dot = cast<DotOperandEncodingAttr>(enc);
  return chooseDotDsReadB64TrLayout(dot, shape, elemBitWidth);
}

LinearLayout chooseScaledMfmaScaleLayout(MLIRContext *ctx, int dotOperandIdx,
                                         ArrayRef<int64_t> dotOperandShape,
                                         unsigned mfmaMDim,
                                         ArrayRef<unsigned> tilesPerWarp,
                                         ArrayRef<unsigned> warpsPerCTA) {
  using basisT = std::vector<std::vector<int32_t>>;
  unsigned rank = dotOperandShape.size();
  auto order = mlir::triton::gpu::getMatrixOrder(rank, /*rowMajor=*/true);
  auto standardOutDims = standardOutDimNames(ctx, rank);
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");

  // Fetch the tilesPerWarp value in the M dimension for operand A, or in the N
  // dimension for operand B.
  unsigned mnDim = dotOperandIdx == 0 ? rank - 2 : rank - 1;
  unsigned tilePerWarpMN = tilesPerWarp[mnDim];

  // In scaled dot, the shapes of operands(without batch dimension) are,
  // respectively:
  // - A: [M, K]
  // - B: [K, N]
  // - aScale: [M, K / 32]
  // - bScale: [N, K / 32]
  //
  // In general, for both 32x32 and 16x16 scaled mfma, and no matter what
  // data type the A/B operand is, each lane takes 32 elements from A/B
  // alone K dim, and 1 or 2 elements from scale accordingly. The number of
  // scale's elements in a lane varies because the 32 elements from A/B may
  // not be consecutive.
  //
  // For mxfp4, these 32 elements are consecutive, so only 1 scale element
  // is required. But for mxfp6/mxfp8, there are 2 16-consecutive elements
  // blocks, so 2 scale elements are required.
  int32_t kSize = dotOperandShape[1];

  std::vector<std::vector<int32_t>> registerBase;
  std::vector<std::vector<int32_t>> laneBase;

  auto threadsInKDim = mfmaMDim == 32 ? 2 : 4;
  for (int32_t elem = threadsInKDim; elem < kSize; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{elem, 0});

  for (int32_t elem = mfmaMDim; elem < tilePerWarpMN * mfmaMDim; elem *= 2)
    registerBase.emplace_back(std::vector<int32_t>{0, elem});

  if (mfmaMDim == 32) {
    // For ROCDL::mfma_scale_f32_32x32x64_f8f6f4 with fp4 input, each lane
    // takes 32 consecutive elements from A alone K dimension. The first
    // 32 lanes collectively handle A[0:32][0:32], and the other 32 lanes
    // collectively handle A[0:32][32:64]. Each lane take 1 scale element
    // accordingly. Similar to B and bScale.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {1, 0}};
  } else {
    assert(mfmaMDim == 16);
    // For ROCDL::mfma_scale_f32_16x16x128_f8f6f4 with fp4 input, each lane
    // takes 32 consecutive elements from A alone K dimension. The first
    // 16 lanes collectively handle A[0:16][0:32], and another 16 lanes
    // collectively handle A[0:16][32:64] and so on. Each lane take 1 scale
    // element accordingly. Similar to B and bScale.
    laneBase = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 0}, {2, 0}};
  }

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);
  LinearLayout tileLayout({{kRegister, registerBase}, {kLane, laneBase}},
                          {outDimNames[order[0]], outDimNames[order[1]]});

  SmallVector<unsigned> warpsPerCTANew =
      (dotOperandIdx == 1)
          ? SmallVector<unsigned>{warpsPerCTA[1], warpsPerCTA[0]}
          : SmallVector<unsigned>{warpsPerCTA[0], warpsPerCTA[1]};

  SmallVector<unsigned> warpOrder = (dotOperandIdx == 1)
                                        ? SmallVector<unsigned>{0, 1}
                                        : SmallVector<unsigned>{1, 0};

  LinearLayout warpLayout =
      identityStandardND(kWarp, warpsPerCTANew, warpOrder);
  LinearLayout ctaLayout = tileLayout.transposeOuts(outDimNames) *
                           warpLayout.transposeOuts(outDimNames);

  auto ctaLay = CTALayoutAttr::get(/*context=*/ctx, /*CTAsPerCGA=*/{1, 1},
                                   /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});
  auto finalLay = combineCtaCgaWithShape(ctaLayout, ctaLay, dotOperandShape);
  return finalLay;
}

std::optional<LinearLayout>
chooseMfmaLikeStoreLayout(RankedTensorType valType) {
  // TODO: WMMA Support on RDNA
  if (!isa<AMDMfmaEncodingAttr>(valType.getEncoding()))
    return {};
  auto mfmaLayout = cast<AMDMfmaEncodingAttr>(valType.getEncoding());

  // We currently only support transposed [B]F16 MFMA32x32 and MFMA16x16 on
  // CDNA4.
  bool isMfma32 = mfmaLayout.getMDim() == 32 && mfmaLayout.getNDim() == 32;
  bool isMfma16 = mfmaLayout.getMDim() == 16 && mfmaLayout.getNDim() == 16;

  auto valShape = valType.getShape();
  // For mfma16x16, to use in-wavefront swap, we need to make sure the tiles
  // used are in one wavefront if there are multiple tiles, which means
  // warpsPerCTA = [numWarps, 1] and at least two tiles along the N dim. For
  // now, it is only possible for FA-like kernels since during mfma generation,
  // the WarpsPerCTA of the head dot in the chain will be reshaped to [numWaprs,
  // 1].
  // TODO: For gemm-like kernel, the transformation here cannot be applied for
  // now and will support it.
  bool validForMfma16 = isMfma16 && valShape.back() >= 16 * 2 &&
                        mfmaLayout.getWarpsPerCTA().back() == 1;

  Type elemType = valType.getElementType();
  if (!(valType.getRank() == 2 && (elemType.isF16() || elemType.isBF16()) &&
        mfmaLayout.getVersion() == 4 && mfmaLayout.getIsTransposed() &&
        (isMfma32 || validForMfma16)))
    return {};

  LinearLayout mfmaLL = mfmaLayout.toLinearLayout(valShape);
  auto mfmaOutDims = llvm::to_vector(mfmaLL.getOutDimNames());
  StringAttr dimM = mfmaOutDims[0];
  StringAttr dimN = mfmaOutDims[1];
  auto swapLL = LinearLayout::empty();
  // The rows are kept as is with an identity linear layout.
  swapLL *= LinearLayout::identity1D(valShape[0], dimM, dimM);
  /*
  clang-format off
  In transposed mfma32 layout, Each thread holds 4 consecutive values along N
  dim. We want to exchange column 4-7 (owned by thread 32-63, BLK0) and column
  8-11 (owned by thread 0-31, BLK1) every 16 columns to make each thread holds 8
  elements. This would mean exchange the 2nd and 3rd basis vector from an
  identity linear layout on tensor elements.

  Correspondingly, the transposed mfma16 layout, the output of
  transposed of mfma16x16 is:

              N/register
  M/Lane          v0       v1       v2       v3       v4       v5       v6       v7
              -------------------------------------------------------------------------
  row0:  0-15 | tile-0 | tile-0 | tile-0 | tile-0 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------
  row1: 16-31 | tile-0 | tile-0 | tile-0 | tile-0 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------
  row2: 32-47 | tile-0 | tile-0 | tile-0 | tile-0 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------
  row3: 48-63 | tile-0 | tile-0 | tile-0 | tile-0 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------
  which means:
  The columns from v0 to v3 are in the one output of mfma16x16 and
  the columns from v4 to v7 are in the one output of mfma16x16,

  The following graph is the same as the one above, execept the tile number is replaced with coordinates in the tenor,
            N/register
            -----------------------------------------------
  M/lane    |(0,  0) ...  (0,  3) | (0,  16) ... (0,  19) |
            |....                 | sub-tensor-0          |
            |(15, 0) ...  (15, 3) | (15, 16) ... (15, 19) |
            -----------------------------------------------
            |(0,  4) ...  (0,  7) | (0,  20) ... (0,  23) |
            |sub-tensor-1         | ....                  |
            |(15, 0) ...  (15, 3) | (15, 20) ... (15, 23) |
            -----------------------------------------------
            |(0,  8) ...  (0,  11)| (0,  24) ... (0,  27) |
            |....                 | sub-tensor-2          |
            |(15, 8) ...  (15, 11)| (15, 24) ... (15, 27) |
            -----------------------------------------------
            |(0,  12) ... (0,  15)| (0,  28) ... (0,  31) |
            |sub-tensor-3         | ....                  |
            |(15, 12) ... (15, 15)| (15, 28) ... (15, 31) |
            -----------------------------------------------
  The basis vector for lane and register are:
  Register = {{0, 1}, {0, 2}}
  Lane = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}, {0, 8}}
  With this layout, only 4xfp16 can be packed in the final global store.

  To use 128-bits global store, we need to pack 8 elements, which means the layout looks like:
              N/register
  M/Lane          v0       v1       v2       v3       v4       v5       v6       v7
              -------------------------------------------------------------------------
  row0:  0-15 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 |
              -------------------------------------------------------------------------
  row1: 16-31 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------
  row2: 32-47 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 | tile-0 |
              -------------------------------------------------------------------------
  row3: 48-63 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 | tile-1 |
              -------------------------------------------------------------------------

  The following graph is the same as the one above, execept the tile number is replaced with coordinates in the tenor:
            N/register
            -----------------------------------------------
            |(0,  0) ...  (0,  3) | (0,  4) ...  (0,  7)  |
            |....                 | sub-tensor-1          |
            |(15, 0) ...  (15, 3) | (15, 16) ... (15, 19) |
            -----------------------------------------------
            |(0, 16) ...  (0, 19) | (0,  20) ... (0,  23) |
            |sub-tensor-0         | ....                  |
            |(15, 16) ... (15, 19)| (15, 20) ... (15, 23) |
            -----------------------------------------------
            |(0,  8) ...  (0,  11)| (0,  12) ... (0,  15) |
            |....                 | sub-tensor-3          |
            |(15, 8) ...  (15, 11)| (15, 12) ... (15, 15) |
            -----------------------------------------------
            |(0,  24) ... (0,  27)| (0,  28) ... (0,  31) |
            |sub-tensor-2         | ....                  |
            |(15, 24) ... (15, 27)| (15, 28) ... (15, 31) |
            -----------------------------------------------
  which means we need to exchange sub-tensor-0 with sub-tensor-1 and sub-tensor-2 and sub-tensor-3.
  And basis vector for lane and register are:
  Register = {{0, 1}, {0, 2}, {0, 4}}
  Lane = {{1, 0}, {2, 0, [4, 0}, {8, 0}, {0, 16}, {0, 8}}

  The steps to get this layout are, firstly we check the last dim of WarpsPerCTA is 1, so we can use v_permlane16.
  Then, we exchange the 2nd and 4th elements in the basis vector of an identity linear and then it will be composed with
  the original mfma16 LL.
            clang-format on
  */
  auto destIdxInBases = isMfma32 ? 3 : 4;
  std::vector<std::vector<int32_t>> dimNBases(mfmaLL.getOutDimSizeLog2(dimN));
  std::generate(dimNBases.begin(), dimNBases.end(),
                [i = 0]() mutable { return std::vector<int32_t>{1 << i++}; });
  std::swap(dimNBases[2], dimNBases[destIdxInBases]);
  swapLL *= LinearLayout({{dimN, dimNBases}}, {dimN});

  return mfmaLL.compose(swapLL);
}

LinearLayout getScaleTMEMStoreLinearLayout(RankedTensorType scaleType,
                                           int numWarps) {
  assert(numWarps == 4 || numWarps == 8);
  MLIRContext *ctx = scaleType.getContext();

  using basisT = std::vector<std::vector<int32_t>>;
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");

  int64_t M = scaleType.getDimSize(0);
  int64_t N = scaleType.getDimSize(1);
  auto CTALayout = getCTALayout(scaleType.getEncoding());
  basisT regBase;

  // Pick a layout that will be trivial to store into the following TMEM layout:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
  // Pack 4 scales together, if there are less than 4 we replicate the data.
  for (int i = 1; i < 4; i = i << 1) {
    if (i >= N)
      regBase.push_back({0, 0});
    else
      regBase.push_back({0, i});
  }
  // Distribute 32 elements of M along a warp.
  basisT laneBase = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}};
  // The data are replicated across all the warps of each warpgroups.
  basisT warpBase = {{0, 0}, {0, 0}};
  for (int i = 32; i < M; i = i << 1) {
    regBase.push_back({i, 0});
  }
  for (int i = 4; i < N; i = i << 1) {
    regBase.push_back({0, i});
  }
  // If we have 8 warps distribute the last dimension on the second warp group.
  if (numWarps == 8) {
    warpBase.push_back(regBase.back());
    regBase.pop_back();
  }

  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, 2);
  auto regLanes =
      LinearLayout({{kRegister, regBase}, {kLane, laneBase}, {kWarp, warpBase}},
                   {outDimNames[0], outDimNames[1]});

  return combineCtaCgaWithShape(regLanes, CTALayout, scaleType.getShape());
}

std::optional<LinearLayout>
getTmemLoadStoreLayout16x256(int M, int N, RankedTensorType oldType,
                             int numWarps) {
  // Too small to distribute on two warp groups while using 16x256 message.
  if (numWarps == 8 && M == 64 && N <= 16 &&
      oldType.getElementTypeBitWidth() < 32) {
    return {};
  }
  assert(numWarps == 4 || numWarps == 8);
  auto ctaLayout = getCTALayout(oldType.getEncoding());
  SmallVector<int64_t> shape = getShapePerCTA(oldType);
  MLIRContext *ctx = ctaLayout.getContext();

  using basisT = std::vector<std::vector<int32_t>>;
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, 2);

  unsigned numElementsPerThread = 256 / oldType.getElementTypeBitWidth();
  int kWidth = 64 / oldType.getElementTypeBitWidth();
  // Follow the layout given by a tmem load using this layout for the inner
  // shape:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-16256b
  LinearLayout innerTile =
      nvidiaMmaTile(ctx, {8, numElementsPerThread}, kWidth, {1, 0}, {0, 1});
  innerTile =
      innerTile * LinearLayout::identity1D(2, kRegister, outDimNames[0]);
  // Then distribute the rest along warpgroups and registers.
  // Then the last warp distribute along M or N following the same order as
  // in getTmemLoadStoreLayout32x32b. This allows us to use the same lowering to
  // tmem for load and store. This part could be generalized by making the
  // lowering of tmem load and store rely more on linear layout.
  bool distributeMAlongWarps = false;
  bool distributeNAlongWarps = false;
  // Figure out how to distribute acorss warpgroups.
  if (numWarps == 8) {
    if (shape[0] > 128) {
      distributeMAlongWarps = true;
    } else {
      distributeNAlongWarps = true;
    }
  }
  int nBase = numElementsPerThread;
  int maxRegN =
      std::min(N, distributeNAlongWarps ? (int)shape[1] / 2 : (int)shape[1]);
  if (maxRegN / nBase > 1) {
    innerTile = innerTile * LinearLayout::identity1D(maxRegN / nBase, kRegister,
                                                     outDimNames[1]);
  }
  if (M != 64) {
    innerTile =
        innerTile * LinearLayout::identity1D(2, kRegister, outDimNames[0]);
  }
  // Distribute M along 4 warps to satisfy TMEM requirements.
  innerTile = innerTile * LinearLayout::identity1D(4, kWarp, outDimNames[0]);

  // Fill out the rest of the shape with M first then N.
  int numMRegDim = std::min(128, (int)shape[0]) / M;
  if (numMRegDim > 1) {
    innerTile = innerTile *
                LinearLayout::identity1D(numMRegDim, kRegister, outDimNames[0]);
  }
  // Dim M=128 should be distributed on the second warp group.
  int nextDim = 128;
  if (distributeMAlongWarps) {
    innerTile = innerTile * LinearLayout::identity1D(2, kWarp, outDimNames[0]);
    nextDim <<= 1;
  }
  numMRegDim = shape[0] / nextDim;
  if (numMRegDim > 1) {
    innerTile = innerTile *
                LinearLayout::identity1D(numMRegDim, kRegister, outDimNames[0]);
  }
  int maxN = distributeNAlongWarps ? shape[1] / 2 : shape[1];
  int numNRegDim = maxN / maxRegN;
  if (numNRegDim > 1) {
    innerTile = innerTile *
                LinearLayout::identity1D(numNRegDim, kRegister, outDimNames[1]);
  }
  if (distributeNAlongWarps) {
    innerTile = innerTile * LinearLayout::identity1D(2, kWarp, outDimNames[1]);
  }
  return combineCtaCgaWithShape(innerTile, ctaLayout, oldType.getShape());
}

LinearLayout getTmemLoadLayoutSplitLongM(int M, int N, RankedTensorType oldType,
                                         int numWarps) {
  assert(numWarps == 8);
  auto ctaLayout = getCTALayout(oldType.getEncoding());
  SmallVector<int64_t> shape = getShapePerCTA(oldType);
  MLIRContext *ctx = ctaLayout.getContext();

  using basisT = std::vector<std::vector<int32_t>>;
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kWarp = StringAttr::get(ctx, "warp");

  // Follow the layout given by a tmem load using this layout:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-fragments-shape-1632b2
  basisT laneBase;
  assert(M == 128);
  for (int i = 1; i < 16; i = i << 1) {
    laneBase.push_back({i, 0});
  }
  basisT regBase;
  for (int i = 1; i < N / 2; i = i << 1) {
    regBase.push_back({0, i});
  }
  laneBase.push_back({0, N / 2});
  // then replicate the pattern.
  for (int i = N; i < shape[1]; i = i << 1) {
    regBase.push_back({0, i});
  }
  for (int i = M; i < shape[0]; i = i << 1) {
    regBase.push_back({i, 0});
  }
  // warp 0 and 4 can only access M[0:32], therefore we need to interleave the
  // data.
  basisT warpBase = {{32, 0}, {64, 0}, {16, 0}};
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, 2);
  auto regLanes =
      LinearLayout({{kRegister, regBase}, {kLane, laneBase}, {kWarp, warpBase}},
                   {outDimNames[0], outDimNames[1]});

  return combineCtaCgaWithShape(regLanes, ctaLayout, oldType.getShape());
}

} // namespace mlir::triton::gpu
