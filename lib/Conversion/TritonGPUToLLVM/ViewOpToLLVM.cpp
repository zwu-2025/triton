#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
namespace {
struct SplatOpConversion : public ConvertOpToLLVMPattern<triton::SplatOp> {
  using ConvertOpToLLVMPattern<triton::SplatOp>::ConvertOpToLLVMPattern;
  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // LLVM::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a LLVM::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  const LLVMTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto tensorTy = cast<RankedTensorType>(resType);
    // Check the converted type for the tensor as depending on the encoding the
    // converter may pick different element types.
    auto srcType = typeConverter->convertType(tensorTy);
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(srcType))
      srcType = structTy.getBody()[0];
    // If the type sizes don't match we need to pack constants.
    if (srcType.isIntOrFloat() && constVal.getType().getIntOrFloatBitWidth() !=
                                      srcType.getIntOrFloatBitWidth()) {
      unsigned cstBitWidth = constVal.getType().getIntOrFloatBitWidth();
      unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();
      assert(cstBitWidth <= srcBitWidth && srcBitWidth % cstBitWidth == 0);
      unsigned ratio = srcBitWidth / cstBitWidth;
      Type intTy = IntegerType::get(elemType.getContext(), cstBitWidth);
      VectorType vecType = VectorType::get(ratio, intTy);
      Value intCst = b.bitcast(constVal, intTy);
      Value vec = b.undef(vecType);
      for (unsigned i = 0; i < ratio; ++i)
        vec = b.insert_element(vecType, vec, intCst, b.int_val(32, i));
      constVal = vec;
    }
    auto llSrc = b.bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, llSrc);
    return packLLElements(loc, typeConverter, elems, rewriter, resType);
  }
  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto typeConverter = getTypeConverter();
    auto llStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                       typeConverter, rewriter, loc);
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

struct UnsplatOpConversion : public ConvertOpToLLVMPattern<triton::UnsplatOp> {
  using ConvertOpToLLVMPattern<triton::UnsplatOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(triton::UnsplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto scrVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    rewriter.replaceOp(op, scrVals[0]);
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!mlir::dyn_cast<SplatElementsAttr>(value))
      return failure();
    auto loc = op->getLoc();
    LLVM::ConstantOp arithConstantOp;
    auto values = mlir::dyn_cast<SplatElementsAttr>(op.getValue());
    auto elemType = values.getElementType();
    Attribute val;
    if (type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs() << "ArithConstantSplatOpConversion get unsupported type: "
                   << value.getType() << "\n";
      return failure();
    }
    // Lower FP8 constant to int8 constant since FP8 types are not supported on
    // LLVM IR.
    if (type::isFloat8(elemType))
      elemType = rewriter.getIntegerType(8);
    auto constOp = rewriter.create<LLVM::ConstantOp>(loc, elemType, val);
    auto typeConverter = getTypeConverter();
    auto llStruct = SplatOpConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, typeConverter, rewriter, loc);
    rewriter.replaceOp(op, llStruct);
    return success();
  }
};

// Convert arith::ConstantOp with an array DenseElementsAttr to a
// LLVM::StructType value.
struct ArithConstantArrayOpConversion
    : public ConvertOpToLLVMPattern<arith::ConstantOp> {
  using ConvertOpToLLVMPattern<arith::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!mlir::dyn_cast<DenseElementsAttr>(value))
      return failure();
    if (mlir::isa<SplatElementsAttr>(value))
      return failure();
    auto tensorTy = cast<RankedTensorType>(op.getType());
    auto loc = op->getLoc();
    auto values = mlir::dyn_cast<DenseElementsAttr>(op.getValue());
    auto elemType = values.getElementType();
    SmallVector<Value> llVals;
    for (auto v : values.getValues<APInt>()) {
      auto ll = rewriter.create<LLVM::ConstantOp>(loc, elemType, v);
      llVals.push_back(ll);
    }
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);

    if (elemsPerThread != llVals.size()) {
      op->emitError(
          "Right now we only support constant arrays with the same number of "
          "elements as the number of threads per warp");
      return failure();
    }
    auto llStruct =
        packLLElements(loc, getTypeConverter(), llVals, rewriter, op.getType());
    rewriter.replaceOp(op, {llStruct});
    return success();
  }
};

struct CatOpConversion : public ConvertOpToLLVMPattern<CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;
  explicit CatOpConversion(LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<CatOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    unsigned elems = getTotalElemsPerThread(resultTy);
    auto typeConverter = getTypeConverter();
    Type elemTy = typeConverter->convertType(resultTy.getElementType());
    SmallVector<Type> types(elems, elemTy);
    // unpack input values
    auto lhsVals = unpackLLElements(loc, adaptor.getLhs(), rewriter);
    auto rhsVals = unpackLLElements(loc, adaptor.getRhs(), rewriter);
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret = packLLElements(loc, typeConverter, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct JoinOpConversion : public ConvertOpToLLVMPattern<JoinOp> {
  using OpAdaptor = typename JoinOp::Adaptor;
  explicit JoinOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<JoinOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The last dimension (the one we're joining) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   2 elements per thread in the last dim.
    //
    // With these invariants, join is trivial: We can count how many contiguous
    // registers belong to the same chunk then we merge the registers between
    // two different chunks.
    Location loc = op->getLoc();
    RankedTensorType dstTy = op.getType();
    auto ll = toLinearLayout(dstTy);
    int splitDim = dstTy.getRank() - 1;
    auto kReg = mlir::StringAttr::get(dstTy.getContext(), "register");
    const auto &bases = ll.getBases();
    const auto &regs = bases.find(kReg)->second;
    int numContiguousValues = 1;
    bool found = false;
    for (const auto &reg : regs) {
      if (reg[splitDim] == 1) {
        found = true;
        break;
      }
      numContiguousValues *= 2;
    }
    assert(found && "Join dimension is not distributed along registers.");
    SmallVector<Value> lhsVals =
        unpackLLElements(loc, adaptor.getLhs(), rewriter);
    SmallVector<Value> rhsVals =
        unpackLLElements(loc, adaptor.getRhs(), rewriter);
    assert(lhsVals.size() == rhsVals.size());
    SmallVector<Value> joinedVals;
    joinedVals.resize(lhsVals.size() * 2);
    for (int i = 0; i < lhsVals.size(); i += numContiguousValues) {
      for (int j = 0; j < numContiguousValues; j++) {
        joinedVals[2 * i + j] = lhsVals[i + j];
        joinedVals[2 * i + numContiguousValues + j] = rhsVals[i + j];
      }
    }
    auto typeConverter = getTypeConverter();
    Value ret = packLLElements(loc, typeConverter, joinedVals, rewriter, dstTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct SplitOpConversion : public ConvertOpToLLVMPattern<SplitOp> {
  using OpAdaptor = typename SplitOp::Adaptor;
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The layout distribute the last dimension along registers
    // - The last dimension (the one we're splitting) has sizePerThread=2,
    // threadPerWarp=1 and warpPerBlock=1.
    //
    // With these invariants, split is trivial: We can count how many contiguous
    // registers belong to the same chunk then we separate the registers between
    // two different chunks.
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto ll = toLinearLayout(srcTy);
    int splitDim = srcTy.getRank() - 1;
    auto kReg = mlir::StringAttr::get(srcTy.getContext(), "register");
    const auto &bases = ll.getBases();
    const auto &regs = bases.find(kReg)->second;
    int numContiguousValues = 1;
    bool found = false;
    for (const auto &reg : regs) {
      if (reg[splitDim] == 1) {
        found = true;
        break;
      }
      numContiguousValues *= 2;
    }
    assert(found && "Split dimension is not distributed along registers.");
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(srcVals.size() % 2 == 0);
    SmallVector<Value> outLhsVals;
    SmallVector<Value> outRhsVals;
    for (int i = 0; i < srcVals.size(); i += 2 * numContiguousValues) {
      for (int j = 0; j < numContiguousValues; j++) {
        outLhsVals.push_back(srcVals[i + j]);
        outRhsVals.push_back(srcVals[i + numContiguousValues + j]);
      }
    }
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Value retLhs =
        packLLElements(loc, typeConverter, outLhsVals, rewriter, resultTy);
    Value retRhs =
        packLLElements(loc, typeConverter, outRhsVals, rewriter, resultTy);
    rewriter.replaceOp(op, {retLhs, retRhs});
    return success();
  }
};
struct ReshapeOpConversion : public ConvertOpToLLVMPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  explicit ReshapeOpConversion(LLVMTypeConverter &typeConverter,
                               PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<ReshapeOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    if (triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType())) {
      return emitOptionalError(loc,
                               "expensive view not supported on reshape op");
    }
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto typeConverter = getTypeConverter();
    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    Value ret = packLLElements(loc, typeConverter, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct ExpandDimsOpConversion : public ConvertOpToLLVMPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  explicit ExpandDimsOpConversion(
      LLVMTypeConverter &typeConverter,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<ExpandDimsOp>(typeConverter, benefit) {}
  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    auto srcVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcLayout = dyn_cast<SliceEncodingAttr>(srcTy.getEncoding());
    if (!srcLayout) {
      return emitOptionalError(
          loc, "ExpandDimsOp only supports SliceEncodingAttr as its input");
    }
    auto resultLayout = resultTy.getEncoding();
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.at(offset));
    }
    Value ret =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};
struct MemDescTransOpConversion
    : public ConvertOpToLLVMPattern<MemDescTransOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MemDescTransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<TensorOrMemDesc>(op.getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                      llvmElemTy, rewriter);
    auto dstSmemObj = SharedMemoryObject(
        srcSmemObj.getBase(), srcSmemObj.getBaseElemType(),
        /*offsets=*/applyPermutation(srcSmemObj.getOffsets(), op.getOrder()));
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct MemDescReshapeOpConversion
    : public ConvertOpToLLVMPattern<MemDescReshapeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MemDescReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<TensorOrMemDesc>(op.getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                      llvmElemTy, rewriter);
    SmallVector<Value> offsets = srcSmemObj.getOffsets();
    // FIXME: This should be done by composing a linear layout with its
    // reshaped counterpart.
    SmallVector<unsigned> srcShape;
    for (int64_t d : op.getSrc().getType().getShape())
      srcShape.push_back(d);
    SmallVector<unsigned> dstShape;
    for (int64_t d : op.getType().getShape())
      dstShape.push_back(d);
    Value linearOffset = LLVM::linearize(rewriter, loc, offsets, srcShape);
    SmallVector<Value> delinearizedOffset =
        LLVM::delinearize(rewriter, loc, linearOffset, dstShape);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto dstSmemObj = SharedMemoryObject(
        srcSmemObj.getBase(), srcSmemObj.getBaseElemType(), delinearizedOffset);
    auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct TransOpConversion : public ConvertOpToLLVMPattern<TransOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // By construction, TransOp::inferReturnTypes ensures that the src encoding
    // is the same as the dst encoding so that this op is a no-op.
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct BroadcastOpConversion
    : public ConvertOpToLLVMPattern<triton::BroadcastOp> {
  using ConvertOpToLLVMPattern<triton::BroadcastOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(result.getType());
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();
    auto typeConverter = getTypeConverter();
    assert(rank == resultTy.getRank());
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals = unpackLLElements(loc, src, rewriter);
    std::map<SmallVector<unsigned>, Value> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }
    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.at(offset));
    }
    Value resultStruct =
        packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct MemDescIndexOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescIndexOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = op->getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());

    // getAllocationShapePerCTA returns the correct number fp4 elements that we
    // need to skip when we have fp4Padded=True. getShapePerCTA does not account
    // for this
    auto stride = product(
        getAllocationShapePerCTA(dstTy.getEncoding(), dstTy.getShape()));
    Value offset = b.mul(op.getIndex(), b.i32_val(stride));
    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    auto base = smemObj.getBase();
    auto elemPtrTy = base.getType();
    auto prevOffsets = smemObj.getOffsets();
    SmallVector<Value> offsetVals(prevOffsets.end() - dstTy.getRank(),
                                  prevOffsets.end());

    // Apply padding based on the amount we move the base ptr
    if (auto padEnc = dyn_cast<PaddedSharedEncodingAttr>(dstTy.getEncoding())) {
      auto bitwidth = dstTy.getElementTypeBitWidth();
      Value padOffset = emitPadding(loc, rewriter, padEnc, bitwidth, offset,
                                    /*offsetInBytes=*/false);
      offset = b.add(offset, padOffset);
    }

    // Advance the pointer and keep the opOffsets as the new shape
    smemObj = SharedMemoryObject(b.gep(elemPtrTy, llvmElemTy, base, offset),
                                 llvmElemTy, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct MemDescSubsliceOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescSubsliceOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescSubsliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescSubsliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = op->getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto destTy = op.getResult().getType();
    auto llvmElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto layoutOrder = getOrder(srcTy);
    auto enc = srcTy.getEncoding();

    auto smemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                   llvmElemTy, rewriter);
    auto opOffsetVals = op.getOffsets();

    auto base = smemObj.getBase();
    auto elemPtrTy = base.getType();
    // Accumulate the logical offsets
    SmallVector<Value> offsetVals;
    for (auto [oldOffVal, opOff] :
         llvm::zip(smemObj.getOffsets(), opOffsetVals)) {
      offsetVals.push_back(b.add(oldOffVal, b.i32_val(opOff)));
    }
    smemObj = SharedMemoryObject(base, llvmElemTy, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct MemDescReinterpretOpConversion
    : public ConvertOpToLLVMPattern<MemDescReinterpretOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MemDescReinterpretOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    MemDescType srcTy = op.getSrc().getType();
    MemDescType dstTy = op.getType();
    Type srcElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    Type dstElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), srcElemTy, b);
    SharedMemoryObject newObj(smemObj.getBase(), dstElemTy, dstTy.getRank(),
                              loc, b);
    b.replaceOp(op, getStructFromSharedMemoryObject(loc, newObj, b));
    return success();
  }
};

} // namespace

void mlir::triton::populateViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReshapeOpConversion>(typeConverter, benefit);
  patterns.add<ExpandDimsOpConversion>(typeConverter, benefit);
  patterns.add<SplatOpConversion>(typeConverter, benefit);
  patterns.add<UnsplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantSplatOpConversion>(typeConverter, benefit);
  patterns.add<ArithConstantArrayOpConversion>(typeConverter, benefit);
  patterns.add<CatOpConversion>(typeConverter, benefit);
  patterns.add<JoinOpConversion>(typeConverter, benefit);
  patterns.add<SplitOpConversion>(typeConverter, benefit);
  patterns.add<MemDescTransOpConversion, MemDescReshapeOpConversion>(
      typeConverter, benefit);
  patterns.add<TransOpConversion>(typeConverter, benefit);
  patterns.add<BroadcastOpConversion>(typeConverter, benefit);
  patterns.add<MemDescSubsliceOpConversion, MemDescIndexOpConversion>(
      typeConverter, benefit);
  patterns.add<MemDescReinterpretOpConversion>(typeConverter, benefit);
}
