# SignFy

A Python-based project designed to bridge communication barriers between hearing and deaf individuals through sign language recognition and translation.

## 🎯 Purpose

SignFy aims to facilitate seamless communication by recognizing and interpreting sign language, making it easier for people to connect and understand each other regardless of their hearing abilities.

## 🛠️ Technology Stack

- **Language**: Python
- **Focus**: Sign Language Recognition & Communication
- **LLM Agent**: Ollama (Local LLM Platform)

## 📚 Data Source

This project utilizes the WLASL (World Level American Sign Language) 2000 dataset from Kaggle to train and validate sign language recognition models.

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Ollama installed ([https://ollama.com/](https://ollama.com/))
- Required dependencies (see requirements.txt)

### Installation

```bash
git clone https://github.com/virwang/SignFy.git
cd SignFy
pip install -r requirements.txt
```

### Setting up Ollama

1. Install Ollama from [https://ollama.com/](https://ollama.com/)
2. Start the Ollama service:
   ```bash
   ollama serve
   ```
3. In another terminal, pull a model (e.g., `llama2`):
   ```bash
   ollama pull llama2
   ```
4. The Ollama API will be available at `http://localhost:11434`

## 📖 Usage

[Add your usage instructions here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues to help improve SignFy.

## 📄 License

[Add your license information here]

## 👥 Support

If you have questions or need assistance, please open an issue on this repository.
