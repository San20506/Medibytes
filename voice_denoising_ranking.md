# Voice Denoising Methods Ranking for Medibytes Integration

**Project Context:** Medibytes is a medical voice-to-discharge-note system that processes doctor voice recordings in noisy hospital environments and converts them into structured ER discharge notes. The system requires real-time processing, HIPAA compliance, and works in both cloud (GPU) and local hospital (CPU) deployment scenarios.

**Ranking Criteria (Medibytes-specific):**
- Medical voice preservation (critical terminology, drug names, vitals)
- Real-time processing for clinical workflow (target <30s total pipeline)
- Privacy/HIPAA compliance (local processing preferred)
- Integration with existing Python pipeline
- Dual deployment: Cloud (GPU) + Local Hospital (CPU)
- Production readiness and maintenance burden
- License compatibility for medical software

---

## 🥇 Tier 1: Best for Medibytes Integration

| Rank | Method | Medibytes Score | Medical Voice Fit | Integration Notes |
|------|--------|-----------------|-------------------|-------------------|
| 1 | **RNNoise (Xiph.Org)** | 9.5/10 | Excellent - preserves speech characteristics | Pure C, BSD license, ~10ms latency, minimal dependencies. Perfect for local hospital CPU deployment and as fallback. Can be integrated into existing `audio_clean.py` |
| 2 | **DeepFilterNet** | 9.0/10 | Very Good - high quality speech enhancement | Rust/Python/ONNX variants, MIT/Apache license. Use ONNX Runtime variant for consistency. Higher quality but more complex. Best for cloud GPU deployment |
| 3 | **Sherpa-ONNX** | 8.5/10 | Good - supports multiple medical speech models | 12 language bindings, ONNX Runtime, cross-platform. Good for multi-language support (en-IN, hi-IN, ta-IN). Can integrate DPDFNet/GT-CRN models |

## 🥈 Tier 2: Good for Specific Medibytes Use Cases

| Rank | Method | Medibytes Score | Medical Voice Fit | Integration Notes |
|------|--------|-----------------|-------------------|-------------------|
| 4 | **DPDFNet** | 8.0/10 | Good - real-time streaming | ONNX/TFLite, ~30ms latency. Can integrate via ONNX Runtime. Good for edge deployment in smaller clinics |
| 5 | **Spectral Subtraction / Wiener Filtering** | 7.0/10 | Basic - simple noise reduction | Public domain, any platform. Good as baseline/fallback for simple noise cases. Minimal integration effort |
| 6 | **ARM CMSIS-DSP** | 6.5/10 | Limited - MCU only | Only if Medibytes targets embedded/MCU platforms for portable devices. Not needed for standard hospital deployment |

## 🥉 Tier 3: Conditional for Medibytes

| Rank | Method | Medibytes Score | Medical Voice Fit | Integration Notes |
|------|--------|-----------------|-------------------|-------------------|
| 7 | **PercepNet** | 6.0/10 | Moderate - perceptual optimization | Unofficial implementation increases risk. Consider if perceptual quality is critical for medical dictation |
| 8 | **NVIDIA NeMo Speech** | 5.5/10 | Good - but GPU-dependent | Only if Medibytes cloud deployment has GPU requirement. Adds heavy dependency, not suitable for local CPU deployment |
| 9 | **WebRTC Audio Processing** | 5.0/10 | Basic - web-focused | Good for web-based Medibytes interface, but less suitable for backend processing |

## ❌ Tier 4: Not Recommended for Medibytes Core

| Rank | Method | Medibytes Score | Medical Voice Fit | Integration Notes |
|------|--------|-----------------|-------------------|-------------------|
| 10 | **SpeechBrain** | 4.0/10 | Research-focused | Too heavy for clinical production, better for research/benchmarking only |
| 11 | **VoiceFixer** | 3.5/10 | Offline restoration | Offline only, different use case than real-time medical dictation |
| 12 | **Microsoft SpeechT5** | 3.0/10 | Research-focused | Heavy PyTorch dependency, overkill for medical voice denoising |
| 13 | **Speex DSP** | 2.5/10 | Historical | Superseded by neural methods, only for legacy compatibility |

---

## 🔧 Recommended Medibytes Integration Architecture

**Core Stack (Tier 1):**
```
Medibytes Voice Denoising Pipeline (Stage 1)
├── RNNoise (baseline/fallback, C implementation)
│   └── Integrated into audio_clean.py for CPU deployment
├── DeepFilterNet-ONNX (primary for cloud GPU)
│   └── Higher quality for noisy hospital environments
└── Sherpa-ONNX (multi-language support)
    └── DPDFNet model for regional language support (hi-IN, ta-IN)
```

**Integration Benefits for Medibytes:**
- **Medical Voice Preservation:** RNNoise preserves speech characteristics critical for drug names, vitals
- **Dual Deployment:** RNNoise for local CPU, DeepFilterNet for cloud GPU
- **HIPAA Compliance:** Local processing with RNNoise, no cloud dependency for sensitive data
- **Pipeline Integration:** All can integrate into existing `audio_clean.py` Stage 1
- **License Safety:** All use permissive licenses (BSD, MIT, Apache 2.0) suitable for medical software
- **Fallback Strategy:** RNNoise as ultra-low-latency fallback (~10ms) for reliable clinical workflow
- **Multi-language:** Sherpa-ONNX supports en-IN, hi-IN, ta-IN for Indian hospital context

**Medibytes-Specific Integration Points:**
- **Current Implementation:** `demo/audio_clean.py` uses basic RMS VAD + optional noisereduce
- **Upgrade Path:** Replace noisereduce with RNNoise for better medical voice preservation
- **Cloud vs Local:** Use RNNoise locally, DeepFilterNet in cloud for quality difference
- **Clinical Workflow:** <10ms RNNoise latency fits within <30s total pipeline target

**Conditional Additions for Medibytes:**
- **Spectral Subtraction:** As simple baseline for basic noise cases
- **NVIDIA NeMo:** Only if Medibytes cloud deployment requires GPU acceleration
- **ARM CMSIS-DSP:** Only if targeting portable medical devices

**Avoid for Medibytes Core:**
- Research-heavy frameworks (SpeechBrain, SpeechT5)
- Offline-only tools (VoiceFixer) 
- Historical libraries (Speex DSP)
- Web-focused solutions (WebRTC) for backend processing

---

## 📊 Medibytes Integration Complexity Matrix

| Method | Medical Voice | CPU Performance | GPU Performance | HIPAA Local | Pipeline Fit | Maintenance |
|--------|---------------|-----------------|-----------------|-------------|--------------|-------------|
| RNNoise | Excellent | Excellent | Good | Excellent | Excellent | Low |
| DeepFilterNet | Very Good | Good | Excellent | Good | Very Good | Medium |
| Sherpa-ONNX | Good | Good | Good | Good | Good | Low |
| DPDFNet | Good | Good | Good | Good | Good | Low |
| Spectral Subtraction | Basic | Excellent | N/A | Excellent | Basic | Very Low |
| ARM CMSIS-DSP | Limited | Excellent | N/A | Excellent | Limited | Low |
| PercepNet | Moderate | Good | Good | Moderate | Moderate | High |
| NVIDIA NeMo | Good | Poor | Excellent | Poor | Moderate | High |
| WebRTC Audio Processing | Basic | Good | N/A | Good | Moderate | Low |
| SpeechBrain | Research | Poor | Good | Poor | Poor | High |
| VoiceFixer | Offline | Poor | Good | Poor | Poor | Medium |
| SpeechT5 | Research | Poor | Good | Poor | Poor | High |
| Speex DSP | Historical | Excellent | N/A | Excellent | Poor | Low |

---

## 🎯 Final Recommendation for Medibytes

**For Medibytes Project Integration:**

1. **Start with RNNoise** - Best balance of medical voice preservation, CPU performance, HIPAA compliance, and simple integration into existing `audio_clean.py`
2. **Add DeepFilterNet-ONNX for cloud** - Higher quality for noisy hospital environments when GPU available
3. **Consider Sherpa-ONNX for multi-language** - If regional language support (hi-IN, ta-IN) is critical for Indian hospitals
4. **Skip Tier 3-4** - Unless specific medical device requirements justify the complexity

**Implementation Priority for Medibytes:**
1. Replace `noisereduce` in `audio_clean.py` with RNNoise C implementation
2. Add DeepFilterNet as optional GPU backend for cloud deployment
3. Implement fallback strategy: RNNoise always available, DeepFilterNet when GPU present
4. Add medical voice preservation testing with drug names, vitals, clinical terminology

This approach provides HIPAA-compliant local processing, medical voice preservation, dual deployment capability, and maintains the <30s clinical pipeline target while avoiding unnecessary complexity.