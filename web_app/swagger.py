from flask import Blueprint, jsonify
from flask_swagger_ui import get_swaggerui_blueprint

swagger_bp = Blueprint('swagger', __name__)

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Auto Video Editor API"
    }
)

# We will serve the swagger.json from the static folder, ensuring it exists there.
