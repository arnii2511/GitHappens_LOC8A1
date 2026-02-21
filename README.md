# Nexport - Swipe to Export

Nexport is an intelligent trade matchmaking system designed to connect exporters with the most relevant international buyers using structured trade data and scoring logic.

Instead of relying on keyword-based search or static directory listings, Nexport evaluates EXIM data, firmographics, and buyer activity signals to generate ranked and explainable match recommendations.

---

## What We Are Building

We are building a data-driven matchmaking engine that:

- Processes trade and company data  
- Detects high-intent buyer signals  
- Generates a Match Score (0–100)  
- Introduces a Trust Score (0–100) to reduce fraud risk  
- Provides ranked recommendations with clear reasoning  

---

## High-Level Architecture

### Frontend (Next.js)
- User interface  
- Authentication  
- Displays match recommendations  

### Node.js API Layer
- Handles API requests  
- Manages user and company data  
- Communicates with Python backend  

### Python Backend
- Data cleaning and normalization  
- Match Score model  
- Trust Score model  
- Explainability logic  

---

Built for hackathon prototype development.
