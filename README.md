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

## Test in postman

```
curl --location 'http://localhost:8000/recommend?productid=<your_product_id>' \
--header 'APIKey: <API_KEY>'
```