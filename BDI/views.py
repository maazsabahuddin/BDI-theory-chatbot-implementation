from flask import jsonify, request, Response
from flask.views import View, MethodView
from flask_api import status
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import jwt
from User.decorators import login_required, password_change_decorator
from .models import BDI
from User.models import UserStats


class GetQuestions(MethodView):

    @login_required
    def get(self, request, data):
        questions = BDI.objects().to_json()
        # user = data.get('user')
        # user_stats = UserStats.objects(user=user).first()
        # print(user_stats.user.email)
        return Response(questions, mimetype="application/json", status=200)

    def post(self):
        return jsonify({
            'message': 'Method \"POST\" not allowed.'
        })
