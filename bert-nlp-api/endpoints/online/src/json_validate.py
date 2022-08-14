from functools import wraps

from flask import jsonify, request
from jsonschema import ValidationError, validate
from werkzeug.exceptions import BadRequest


def validate_json(f):
    @wraps(f)
    def wrapper(*args, **kw):
        # リクエストのコンテンツタイプがjsonかどうかをチェック
        ctype = request.headers.get("Content-Type")
        method_ = request.headers.get("X-HTTP-Method-Override", request.method)
        if method_.lower() == request.method.lower() and "json" in ctype:
            try:
                # bodyメッセージの有無をチェック
                request.json
            except BadRequest as e:
                msg = "This is an invalid json"
                return jsonify({"error": msg}), 400
            return f(*args, **kw)

    return wrapper


def validate_schema(schema):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            try:
                # JSON-Schemaの定義通りかをチェック
                validate(request.json, schema)
            except ValidationError as e:
                # return jsonify({"error": e.message}), 400
                return jsonify({"error": e}), 400
            return f(*args, **kw)

        return wrapper

    return decorator