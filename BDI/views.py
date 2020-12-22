from flask import jsonify, request, Response
from flask.views import View, MethodView
from flask_api import status
from werkzeug.security import generate_password_hash, check_password_hash
import datetime
import jwt

from Main.settings.production import NOT_CATCHABLE_ERROR_CODE, NOT_CATCHABLE_ERROR_MESSAGE
from User.decorators import login_required, password_change_decorator
from .models import BDI
from User.models import UserStats
FINAL_QUESTION = 21

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


class UserStatistics(MethodView):

    @login_required
    def get(self, request, data):
        try:
            user = data.get('user')
            user_stats = UserStats.objects(user=user).first()
            if not user_stats:
                return jsonify({
                    'status': status.HTTP_200_OK,
                    'score': 0,
                    'message': 'Successful'
                })

            answered_question = user_stats.question_answered
            question_no = answered_question[-1].get('question_no')
            if question_no == FINAL_QUESTION:
                return jsonify({
                    'status': status.HTTP_200_OK,
                    'score': user_stats.score,
                    'message': 'Successful'
                })

            return jsonify({
                'status': status.HTTP_200_OK,
                'score': 0,
                'message': 'Successful'
            })

        except Exception as e:
            return jsonify({'status': NOT_CATCHABLE_ERROR_CODE, 'message': NOT_CATCHABLE_ERROR_MESSAGE})

    def post(self):
        return jsonify({
            'message': 'Method \"POST\" not allowed.'
        })