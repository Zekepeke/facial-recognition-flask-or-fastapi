# facial-recognition-flask-or-fastapi
A project to test how a server could use Facial Detection and Recognition and send the data to a website using flask or fastapi(haven't made a choice yet).


## Setup

```bash
# Clone the repo
git clone https://github.com/Zekepeke/facial-recognition-flask-or-fastapi.git
cd facial-recognition-flask-or-fastapi
```

## Backend

make sure to have python 3.11 💔

```bash
# from server directory
python3.11 -m venv facial-project
# macOS/Linux
source facial-project/bin/activate
# Windows (PowerShell)
# facial-project\Scripts\Activate.ps1

python -m pip install --upgrade pip

# App dependencies
pip install -r requirements.txt

# run (pick one)
export FLASK_APP=app.py FLASK_ENV=development && flask run --port 8000
# or
python app.py
```

## Trying out the Backend

In a separate do this and make sure to have images(try obama if you want)
```bash
curl -F "file=@images/obama.jpeg" http://localhost:8000/enroll/obama
curl -F "file=@images/obama1.jpeg" http://localhost:8000/enroll/obama
curl -F "file=@images/obama2.jpeg" http://localhost:8000/enroll/obama
curl -F "file=@images/obama3.jpeg" http://localhost:8000/enroll/obama


# identify (same or different pic)
curl -F "file=@images/obama4.jpeg" -F "threshold=0.65" http://localhost:8000/identify
```
## Frontend (Vite + React)

Install deps, configure the API base URL, and start the dev server.

```bash
# from your frontend directory
npm install
```

Create .env.local:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

Run:

```bash
npm run dev
# usually serves at http://localhost:5173
```
