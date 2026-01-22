# Artificial Intelligence model to assign treatment to a sick person based on a clinical note

## 🎯 Project Idea

This project implements an **automatic medical specialty classification system** using advanced **Natural Language Processing (NLP)** and **Artificial Intelligence** techniques. The main objective is to analyze clinical histories in text format and predict their corresponding medical specialty, facilitating the classification and organization of medical records.

### Innovative Concept

The system combines the power of **BioBERT** (a pre-trained language model on biomedical literature) with **classical Machine Learning** algorithms to create a robust classifier that:

- 📋 Processes complete medical transcriptions
- 🧬 Extracts semantic features from medical language
- 🎯 Predicts medical specialty with confidence levels
- 📊 Can classify among multiple specialties simultaneously

## 📚 Technical Description

### Technologies Used

- **BioBERT v1.1**: Transformer model specialized in biomedical domain
- **PyHealth**: Framework for healthcare data analysis
- **Scikit-learn**: Machine learning model implementation
- **PyTorch**: Deep learning framework
- **Transformers (Hugging Face)**: Library for pre-trained language models
- **Pandas & NumPy**: Data manipulation and analysis

### System Architecture

```
Clinical History (Text)
        ↓
BioBERT Tokenizer → Embeddings (768 dimensions)
        ↓
Logistic Regression → Multi-class Classification
        ↓
Prediction + Probabilities per Specialty
```

### Development Process

#### 1. **Data Acquisition**
- Dataset: Medical Transcriptions (Kaggle)
- Source: `mtsamples.csv`
- Content: Real medical transcriptions with labeled specialties

#### 2. **Preprocessing**
- Null data cleaning
- Selection of the 5 most frequent specialties
- Filtering of complete clinical histories

#### 3. **Embedding Generation**
- Tokenization with BioBERT tokenizer
- Maximum length: 128 tokens
- Extraction of 768-dimensional vector per document

#### 4. **Model Training**
- Algorithm: Logistic Regression with `class_weight='balanced'`
- Split: 80% training, 20% testing
- Label encoding with `LabelEncoder`

#### 5. **Evaluation and Prediction**
- Metrics: Precision, Recall, F1-Score
- Prediction with confidence levels per category

## 🚀 Installation and Usage

### Prerequisites

```bash
pip install pyhealth "pandas==1.5.3" "numpy<2.0.0"
pip install transformers torch
```

### Execution

1. Clone the repository:
```bash
git clone https://github.com/tulio3101/PTIA.git
cd PTIA
```

2. Open the notebook:
```bash
jupyter notebook PTIA_PROYECTO_FINAL.ipynb
```

3. Execute cells sequentially to:
   - Download the dataset automatically
   - Train the model
   - Make predictions

## 📊 Results

The model is capable of classifying clinical histories into the following main specialties:

- **Surgery**
- **Orthopedic**
- **Cardiovascular / Pulmonary**
- **Radiology**
- **Gastroenterology**

### Prediction Example

```python
# Sample medical note
note = """CHIEF COMPLAINT: Shortness of breath and palpitations.
HISTORY: Patient with coronary artery disease.
IMPRESSION: Atrial Fibrillation, Heart Failure."""

# System predicts: Cardiovascular / Pulmonary (with % confidence)
```

## 🎓 Academic Context

**Institution**: Escuela Colombiana de Ingeniería Julio Garavito  
**Course**: Principles and Technologies of Artificial Intelligence  
**Period**: 2025-2  
**Type**: Final Project

### Achieved Objectives

✅ Develop an AI model for disease prediction  
✅ Apply preprocessing and data cleaning techniques  
✅ Implement machine learning with clinical data  
✅ Evaluate and validate the model with standard metrics  
✅ Reinforce theoretical and practical concepts from the course

## 👥 Contributors

- **[@tulio3101](https://github.com/tulio3101)** - Main development
- **[@sebasPuentes](https://github.com/sebasPuentes)** - Project collaborator

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [BioBERT Paper](https://arxiv.org/abs/1901.08746)
- [Medical Transcriptions Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- [PyHealth Documentation](https://pyhealth.readthedocs.io/)

---

**Note**: This project is for academic and research purposes. It should not be used for actual medical diagnoses without proper professional supervision.
