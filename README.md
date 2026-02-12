## 📄 Paper

This repository contains the official implementation of our Interspeech 2025 paper:

**Pei-Chin Hsieh et al., "Tonality-Based Accompaniment-Guided Automatic Singing Evaluation," Interspeech 2025.**

🔗 Paper: https://www.isca-archive.org/interspeech_2025/hsieh25c_interspeech.pdf

## 📖 Citation

If you use this repository, please cite:

```bibtex
@inproceedings{hsieh2025interspeech,
  author    = {Pei-Chin Hsieh, Yih-Liang Shen, Ngoc-Son Tran, and Tai-Shih Chi},
  title     = {Tonality-Based Accompaniment-Guided Automatic Singing Evaluation},
  booktitle = {Proc. Interspeech 2025},
  year      = {2025}
}
```

## Quick Start 
1. Objective Score  
   (Section 3.1: Tonality-Based Singing Assessment)  
   To see how objective score is calculated:
   ```bash
   tonality_scoring.ipynb
   ```
   (Section 3.2: Pitch Extraction)  
   The pitch arrays were extracted using our own model/YIN algorithm and stored as .txt files for convenience, you can apply any pitch extraction model/algorithm with higher accuracy.  

   (Section 3.3: Musical Key Classification)  
   To run tonality model inference:
    ```bash
   $ python model_test/onnx_test.py
   ```
   
2. Subjective Score  
   (Section 4.1: Subjective Rating)  
   Run the following script to reproduce our subjective rating result:
   ```bash
   $ python listening_test/subjective_score.py
   
3. Correlation Analysis  
  (Section 4.2: Experiment Results)  
  To compute correlation between objective score & human rating distribution, run:
   ```bash
   tonality_correlation_quantize.ipynb
   ```
