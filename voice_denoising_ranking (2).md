# Voice Denoising Methods Ranking for Medibyte Integration

**Ranking Criteria:** Integration suitability for a single project (Medibyte), considering:
- Integration complexity and codebase cohesion
- License compatibility 
- Platform overlap and deployment flexibility
- Maintenance burden
- Performance/latency trade-offs
- Production readiness

---

## 🥇 Tier 1: Best for Unified Integration

| Rank | Method | Integration Score | Key Strengths | Integration Notes |
|------|--------|-------------------|---------------|-------------------|
| 1 | **Sherpa-ONNX** | 9.5/10 | 12 language bindings, ONNX Runtime, cross-platform, streaming API | Best integration flexibility - single ONNX Runtime backend supports multiple models (DPDFNet, GT-CRN), unified API across platforms |
| 2 | **RNNoise (Xiph.Org)** | 9.0/10 | Pure C, BSD license, ~10ms latency, minimal dependencies | Easiest C integration, lightweight, can be baseline or fallback option |
| 3 | **DeepFilterNet** | 8.5/10 | High quality, Rust/Python/ONNX variants, MIT/Apache license | Use ONNX Runtime variant for consistency with Sherpa-ONNX, higher quality but more complex |

## 🥈 Tier 2: Good Integration with Specific Use Cases

| Rank | Method | Integration Score | Key Strengths | Integration Notes |
|------|--------|-------------------|---------------|-------------------|
| 4 | **DPDFNet** | 8.0/10 | ONNX/TFLite, ~30ms latency, stateful streaming | Can integrate via ONNX Runtime alongside Sherpa-ONNX models |
| 5 | **ARM CMSIS-DSP** | 7.5/10 | Apache 2.0, optimized for ARM Cortex-M | Only if Medibyte targets embedded/MCU platforms, otherwise not needed |
| 6 | **Spectral Subtraction / Wiener Filtering** | 7.0/10 | Public domain, any platform, very simple | Good as baseline/fallback, minimal integration effort |

## 🥉 Tier 3: Conditional Integration

| Rank | Method | Integration Score | Key Strengths | Integration Notes |
|------|--------|-------------------|---------------|-------------------|
| 7 | **PercepNet** | 6.5/10 | BSD-3-Clause, perceptual quality, ~30ms | Unofficial implementation increases integration risk, consider if perceptual optimization is critical |
| 8 | **NVIDIA NeMo Speech** | 6.0/10 | GPU-optimized, Apache 2.0, production-ready | Only if Medibyte has GPU deployment requirement, adds heavy dependency |
| 9 | **Speex DSP** | 5.5/10 | C, BSD, historical, proven | Superseded by neural methods, only for legacy compatibility |

## ❌ Tier 4: Not Recommended for Core Integration

| Rank | Method | Integration Score | Key Strengths | Integration Notes |
|------|--------|-------------------|---------------|-------------------|
| 10 | **SpeechBrain** | 4.0/10 | Research toolkit, many models | Too heavy for production, better for research/benchmarking only |
| 11 | **VoiceFixer** | 3.5/10 | Offline restoration, MIT | Offline only, different use case than real-time denoising |
| 12 | **Microsoft SpeechT5** | 3.0/10 | Transformer-based, MIT | Research-focused, heavy PyTorch dependency, overkill for denoising |

---

## 🔧 Recommended Integration Architecture

**Core Stack (Tier 1):**
```
Medibyte Voice Denoising Engine
├── Sherpa-ONNX (primary, via ONNX Runtime)
│   ├── DPDFNet model (default)
│   └── GT-CRN model (alternative)
├── RNNoise (baseline/fallback, C implementation)
└── DeepFilterNet-ONNX (high-quality option)
```

**Integration Benefits:**
- **Unified Backend:** All Tier 1 methods can run via ONNX Runtime (except RNNoise C)
- **License Compatibility:** All use permissive licenses (BSD, MIT, Apache 2.0)
- **Platform Coverage:** Desktop, mobile, embedded, web via ONNX Runtime
- **Fallback Strategy:** RNNoise as ultra-low-latency fallback (~10ms)
- **Quality Options:** Multiple models for different quality/latency trade-offs

**Conditional Additions:**
- **ARM CMSIS-DSP:** Only if targeting ARM Cortex-M microcontrollers
- **Spectral Subtraction:** As simple baseline implementation
- **NVIDIA NeMo:** Only if GPU acceleration is required

**Avoid for Core:**
- Research-heavy frameworks (SpeechBrain, SpeechT5)
- Offline-only tools (VoiceFixer)
- Historical libraries (Speex DSP)

---

## 📊 Integration Complexity Matrix

| Method | Dependencies | API Complexity | Platform Support | Maintenance |
|--------|-------------|----------------|------------------|-------------|
| Sherpa-ONNX | ONNX Runtime | Medium | Excellent | Low |
| RNNoise | None | Low | Good | Low |
| DeepFilterNet | PyTorch/ONNX | Medium | Good | Medium |
| DPDFNet | ONNX Runtime | Low | Good | Low |
| ARM CMSIS-DSP | None | Low | ARM only | Low |
| Spectral Subtraction | None | Very Low | Excellent | Very Low |
| PercepNet | Custom | Medium | C/C++ only | High (unofficial) |
| NVIDIA NeMo | PyTorch, CUDA | High | GPU only | High |
| Speex DSP | None | Low | Good | Low (legacy) |
| SpeechBrain | PyTorch, many deps | High | Good | High |
| VoiceFixer | PyTorch | Medium | Good | Medium |
| SpeechT5 | PyTorch, HuggingFace | High | Good | High |

---

## 🎯 Final Recommendation

**For Medibyte Project Integration:**

1. **Start with Sherpa-ONNX** - Best balance of integration flexibility, performance, and maintenance
2. **Add RNNoise as fallback** - Ultra-low latency option with minimal integration effort
3. **Consider DeepFilterNet-ONNX** - If higher quality is needed and latency budget permits
4. **Skip Tier 3-4** - Unless specific requirements justify the complexity

This approach provides a unified, maintainable codebase with multiple quality/latency options while avoiding unnecessary complexity.