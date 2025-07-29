## Conda env

```
python -m venv venv
source venv/bin/activate
```

## Install Dependencies
```
pip install -r requirements.txt
```

## Run the application

```
uvicorn main:app --reload --port 8000
```