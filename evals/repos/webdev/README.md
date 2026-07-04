# TinyServe

A lightweight web application framework for building HTTP APIs and server-rendered
websites in Python. TinyServe gives you routing, request/response handling,
templating, and a development server with hot reload — with no heavy dependencies.

## Features

- **Routing** — decorator-based URL routing with path parameters and HTTP method
  dispatch.
- **Templating** — Jinja2-powered HTML rendering with layout inheritance.
- **Request handling** — form parsing, JSON bodies, cookies, and sessions.
- **Middleware** — pluggable request/response middleware for auth, CORS, and
  logging.
- **ORM integration** — works with SQLAlchemy for database-backed apps.
- **WSGI** — deploy behind gunicorn or any WSGI server.

## Quick start

```python
from tinyserve import App

app = App()

@app.route("/hello/<name>")
def hello(request, name):
    return app.render("hello.html", name=name)

app.run(port=8000)
```

TinyServe is for web developers building REST APIs and websites — no machine
learning required.
