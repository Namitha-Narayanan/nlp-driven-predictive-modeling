# API_README File — Stateless Prediction API

## 1. General Approach & Assumptions
This API predicts unseen targets \(y_{n+1:n+k}\) from observed data and a natural-language description `t`.  
My implementation:
- **Hint parsing**: `t` is parsed into a structured spec (e.g. linear, quadratic, periodic).  
- **Basis construction**: polynomial and sinusoidal features are built according to the spec.  
- **Model fitting**: ridge regression is used for robustness with small datasets.  
- **Prediction**: the fitted model is applied to `x_predict` to generate deterministic outputs.  

**Assumptions**:
- Text hints reliably describe the functional form.  
- Relationships can be captured with polynomial + sinusoidal bases.  
- Noise is moderate and Gaussian, so ridge regression is suitable.  

---

## 2. Tools Used
- **Flask + flask-cors** → REST API server with CORS enabled for frontend testing.  
- **NumPy, SciPy** → array handling and linear algebra.  
- **Scikit-learn** → ridge regression model.  
- **Matplotlib, Seaborn** → used during development for quick checks.  
- **GenAI tools**: GitHub Copilot and ChatGPT assisted in scaffolding boilerplate code and documentation. Final design and logic were implemented manually.

---

## 3. Code Structure & Usage
**Structure**
- `main.py` → entry point, creates Flask app and registers routes.  
- `app/routes/` → API endpoints:  
  - `health.py` → `GET /health` returns `{ "status": "ok" }`.  
  - `predict.py` → `POST /predict` validates input, parses hint, builds basis, fits model, returns predictions.  
- `app/services/` → core ML logic:  
  - `text_hints.py` parses natural language into feature specs.  
  - `basis.py` builds design matrices.  
  - `model.py` handles ridge regression fit and predict.  
- `app/utils/` → validation, response helpers, array utilities, config.  
- `static/` → development payloads, ground truth values, simple frontend.  
- `test_stateless_api.py` → smoke test for health, predictions, and error handling.  
- `docs/screenshots`- Stores the test evidence

**Usage**
```bash
pip install -r requirements.txt
python main.py
# API runs at http://127.0.0.1:5000
```
**Testing**
```bash
python test_stateless_api.py
```
Runs 3 checks: health, valid prediction, and error handling.

## 4. Optional Frontend

For convenience, a minimal frontend is included at `static/frontend.html`.  
- Start the backend API (`python main.py`).  
- In a separate terminal:  
```bash
  cd static
  python -m http.server 5173
```
- Open http://localhost:5173/frontend.html in your browser.

This interface allows loading development payloads, sending prediction requests, and viewing outputs (including RMSE when comparing with y_true).

## 5. Other Important Notes
- **Stateless design** → each request contains all inputs; no state is stored between requests.
- **Error handling** → invalid or malformed inputs return JSON of the form:
```bash
{ "status": "error", "message": "description of issue" }
```
Ensuring clarity for both users and automated evaluation.
- **Performance** → predictions are deterministic and efficient, running comfortably within the evaluation time limits.
- **Limitations** → currently supports polynomial and sinusoidal bases only; does not provide uncertainty intervals.
- **Future directions** → extend hint parsing to support interactions and piecewise functions, add automatic periodicity detection, and introduce prediction intervals for more informative outputs.