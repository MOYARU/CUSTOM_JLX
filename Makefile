CXX      = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 \
           -Iinclude -Iinclude/runtime \
           -fobjc-arc
LDFLAGS  = -framework Metal -framework Foundation -framework MetalPerformanceShaders

SRC_DIR   = src
BUILD_DIR = build
TARGET    = custom_jlx
INFER     = jlx_inference

VPATH = $(SRC_DIR):$(SRC_DIR)/runtime

# Shared object files (used by both train and inference)
SHARED_OBJS = $(BUILD_DIR)/Tensor.o \
              $(BUILD_DIR)/Kernel.o \
              $(BUILD_DIR)/Layer.o \
              $(BUILD_DIR)/SparseTernaryLinear.o \
              $(BUILD_DIR)/SparseAttention.o \
              $(BUILD_DIR)/SparseFFN.o \
              $(BUILD_DIR)/Config.o

# Training-only objects
TRAIN_OBJS  = $(BUILD_DIR)/main.o \
              $(BUILD_DIR)/DataLoader.o \
              $(BUILD_DIR)/Adafactor.o \
              $(BUILD_DIR)/TierManager.o

# Inference-only objects
INFER_OBJS  = $(BUILD_DIR)/inference.o \
              $(BUILD_DIR)/Adafactor.o

all: $(TARGET) $(INFER)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SHARED_OBJS) $(TRAIN_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(INFER): $(SHARED_OBJS) $(INFER_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.mm | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(INFER)

run: $(TARGET)
	./$(TARGET)

inference: $(INFER)

.PHONY: all clean run inference