# GPU-Focused Voice Denoising Strategy for Medibytes

**GPU-First Approach:** Medibytes production deployment uses GPU workers (A10G) in cloud EKS cluster. This analysis focuses on GPU-optimized methods that can be layered for superior medical voice enhancement.

---

## 🎯 GPU-Optimized Methods for Layered Processing

### **Tier 1: GPU-Native Methods (Best for Sequential Layering)**

| Method | GPU Optimization | Medical Voice Fit | Latency | Layer Position |
|--------|-----------------|-------------------|---------|----------------|
| **DeepFilterNet** | Native CUDA/ONNX GPU | Excellent | ~40ms | Primary layer |
| **NVIDIA Audio Effects SDK (Maxine)** | Tensor cores, GPU-optimized | Excellent | <30ms | Primary/Secondary |
| **NVIDIA NeMo Speech** | GPU-accelerated inference | Very Good | Variable | Secondary layer |
| **SpeechBrain (SepFormer)** | PyTorch GPU | Good | Higher | Tertiary layer |

### **Tier 2: Hybrid GPU/CPU Methods**

| Method | GPU Usage | Medical Voice Fit | Latency | Layer Position |
|--------|-----------|-------------------|---------|----------------|
| **Sherpa-ONNX (CUDA backend)** | ONNX Runtime GPU | Good | ~30ms | Primary/Secondary |
| **DPDFNet (ONNX GPU)** | ONNX Runtime GPU | Good | ~30ms | Secondary layer |
| **RNNoise (fallback)** | CPU-only | Excellent | ~10ms | Fallback layer |

---

## 🔄 Sequential Layering Strategies

### **Strategy A: Progressive Enhancement Pipeline**

```
Raw Audio → FFmpeg Normalization → DeepFilterNet → NVIDIA Maxine → Medical Speech Enhancement
```

**Layer 1: FFmpeg + Loudnorm (CPU)**
- **Purpose:** Standardize format, normalize volume
- **Cost:** Minimal CPU overhead
- **Medical Benefit:** Consistent input for downstream models

**Layer 2: DeepFilterNet (GPU)**
- **Purpose:** Broad-spectrum noise reduction
- **GPU Cost:** ~0.5-1.0 GB VRAM, ~40ms latency
- **Medical Benefit:** Preserves speech characteristics while removing hospital noise

**Layer 3: NVIDIA Audio Effects SDK (GPU)**
- **Purpose:** Studio-quality enhancement, dereverberation
- **GPU Cost:** ~1-2 GB VRAM, ~30ms latency
- **Medical Benefit:** Enhanced clarity for drug names, clinical terminology

**Layer 4: Medical Speech Enhancement (Custom GPU)**
- **Purpose:** Medical terminology preservation
- **GPU Cost:** ~0.5-1.0 GB VRAM, ~20ms latency
- **Medical Benefit:** Optimized for vitals, drug names, medical phrases

**Total Pipeline Cost:** ~2-4 GB VRAM, ~90-120ms latency
**Medical Voice Quality:** Excellent (multi-stage enhancement)

### **Strategy B: Adaptive Quality Pipeline**

```
Raw Audio → Noise Classification → [Low Noise: RNNoise] | [High Noise: DeepFilterNet + Maxine]
```

**Layer 1: Noise Classification (Lightweight GPU)**
- **Purpose:** Assess noise level, choose appropriate enhancement path
- **GPU Cost:** ~0.1 GB VRAM, ~5ms latency
- **Medical Benefit:** Avoids over-processing clean audio

**Path A (Low Noise): RNNoise (CPU)**
- **Purpose:** Light enhancement for already-clear audio
- **Cost:** CPU only, ~10ms latency
- **Medical Benefit:** Faster processing, minimal artifacts

**Path B (High Noise): DeepFilterNet + Maxine (GPU)**
- **Purpose:** Heavy enhancement for noisy hospital environments
- **GPU Cost:** ~2-3 GB VRAM, ~70ms latency
- **Medical Benefit:** Maximum clarity for difficult audio

**Total Pipeline Cost:** ~0.1-3 GB VRAM, ~15-75ms latency (adaptive)
**Medical Voice Quality:** Very Good (adaptive to input quality)

---

## 🧩 Core Model Block vs Pipeline Approaches

### **Pipeline Approach (Recommended for Medibytes)**

**Architecture:**
```
Stage 1 Worker (GPU):
├── FFmpeg Normalization (CPU)
├── Noise Classification (GPU)
├── Conditional Enhancement (GPU/CPU)
└── Medical Quality Check (GPU)
```

**Advantages:**
- **Modular:** Each layer can be updated independently
- **Debuggable:** Can inspect intermediate results
- **Adaptive:** Can route based on audio quality
- **Resilient:** Single layer failure doesn't break entire pipeline
- **Cost-Optimized:** Use GPU only when needed

**Disadvantages:**
- **Higher Latency:** Sequential processing adds latency
- **Complexity:** More moving parts to maintain
- **Memory Management:** Need to manage intermediate buffers

### **Core Model Block Approach**

**Architecture:**
```
Single GPU Model:
├── Input: Raw Audio
├── Internal: Multi-stage neural network
└── Output: Enhanced Audio
```

**Advantages:**
- **Lower Latency:** Single forward pass
- **Simpler Deployment:** One model to manage
- **End-to-End Optimization:** Joint training of all stages

**Disadvantages:**
- **Black Box:** Hard to debug intermediate results
- **Rigid:** Can't adapt to different input qualities
- **Training Complexity:** Need large medical voice dataset
- **Update Risk:** Changing one aspect requires retraining entire model

---

## 💰 Cost Analysis

### **GPU Resource Costs (Per Worker)**

| Approach | VRAM Usage | GPU Utilization | Hourly Cost (A10G) | Throughput |
|----------|------------|-----------------|-------------------|------------|
| **Single DeepFilterNet** | 1-2 GB | 40-60% | $1.50-2.00 | 60-90 jobs/hour |
| **Layered Pipeline** | 2-4 GB | 70-90% | $2.50-3.50 | 40-60 jobs/hour |
| **Adaptive Pipeline** | 0.5-3 GB | 30-80% | $1.00-3.00 | 50-80 jobs/hour |
| **Core Model Block** | 3-5 GB | 80-95% | $3.00-4.00 | 30-50 jobs/hour |

### **Development & Maintenance Costs**

| Approach | Development Time | Maintenance | Updates | Debugging |
|----------|------------------|-------------|---------|-----------|
| **Single Model** | 6-12 months | Medium | High (retrain) | Difficult |
| **Pipeline** | 3-6 months | Low | Low (swap layers) | Easy |
| **Adaptive** | 4-8 months | Medium | Medium | Medium |

---

## 🔬 Individual Method Differences

### **DeepFilterNet vs NVIDIA Maxine**

| Aspect | DeepFilterNet | NVIDIA Maxine |
|--------|---------------|---------------|
| **Architecture** | Two-stage deep filtering | Studio-quality AI processing |
| **Medical Voice** | Very Good preservation | Excellent clarity |
| **GPU Optimization** | ONNX Runtime/CUDA | Tensor cores optimized |
| **Latency** | ~40ms | <30ms |
| **Cost** | Open source (free) | Commercial SDK |
| **Integration** | Python/Rust/ONNX | SDK/API |
| **Best For** | General hospital noise | Professional audio quality |

### **Sequential Layering Benefits**

**DeepFilterNet → NVIDIA Maxine:**
- **DeepFilterNet:** Removes broad spectrum noise (hospital machinery, background chatter)
- **NVIDIA Maxine:** Adds studio-quality enhancement, dereverberation
- **Combined:** Maximum clarity for medical terminology

**Cost:** ~2-3 GB VRAM, ~70ms total latency
**Medical Voice Quality:** Excellent

### **Sherpa-ONNX → Medical Enhancement**

**Sherpa-ONNX:** Multi-language support (en-IN, hi-IN, ta-IN)
**Medical Enhancement:** Custom model for medical terminology
**Combined:** Language-specific medical voice enhancement

**Cost:** ~1-2 GB VRAM, ~50ms total latency
**Medical Voice Quality:** Very Good (multi-language)

---

## 🎯 Recommended GPU Strategy for Medibytes

### **Primary Recommendation: Adaptive Layered Pipeline**

```
Stage 1 GPU Worker:
├── FFmpeg Normalization (CPU)
├── Noise Classification (Lightweight GPU)
├── [Low Noise] → RNNoise (CPU)
├── [High Noise] → DeepFilterNet + NVIDIA Maxine (GPU)
└── Medical Quality Check (GPU)
```

**Rationale:**
- **Cost-Optimized:** Use GPU only when needed
- **Medical Voice Quality:** Excellent for noisy environments
- **Latency:** Adaptive (15-75ms based on input)
- **Resilient:** Fallback to CPU if GPU unavailable
- **Maintainable:** Modular layers, easy updates

### **Implementation Priority:**

1. **Phase 1:** Implement DeepFilterNet as primary GPU layer
2. **Phase 2:** Add noise classification for adaptive routing
3. **Phase 3:** Integrate NVIDIA Maxine for high-noise cases
4. **Phase 4:** Develop medical-specific enhancement layer

### **Cost Projection:**

**Initial Deployment (Phase 1):**
- **GPU Workers:** 2-4 A10G instances
- **Hourly Cost:** $3.00-8.00
- **Throughput:** 120-360 jobs/hour
- **Medical Voice Quality:** Very Good

**Full Deployment (Phase 4):**
- **GPU Workers:** 4-8 A10G instances
- **Hourly Cost:** $6.00-16.00
- **Throughput:** 200-480 jobs/hour
- **Medical Voice Quality:** Excellent

---

## 📊 Performance Comparison

| Approach | Medical Voice Quality | Latency | GPU Cost | Throughput | Maintenance |
|----------|----------------------|---------|----------|------------|-------------|
| **Single DeepFilterNet** | Very Good | 40ms | Low | High | Low |
| **Layered Pipeline** | Excellent | 90ms | Medium | Medium | Low |
| **Adaptive Pipeline** | Excellent | 15-75ms | Low-Medium | High | Medium |
| **Core Model Block** | Excellent | 50ms | High | Low | High |

**Winner:** **Adaptive Layered Pipeline** - Best balance of medical voice quality, cost, and maintainability for Medibytes.