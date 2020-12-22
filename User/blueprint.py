from flask import Blueprint
from User.views import GetUser, Register, Login, Logout, VerifyUser, UserDetails, UpdateName, ChangePassword, \
    UpdateEmail, ResetPassword

my_view = Blueprint('my_view', __name__)
my_view.add_url_rule("/login/", view_func=Login.as_view('login_view'))
my_view.add_url_rule("/get/users/", view_func=GetUser.as_view('get_users'))
my_view.add_url_rule("/user/details/", view_func=UserDetails.as_view('user_details'))

my_view.add_url_rule("/update/name/", view_func=UpdateName.as_view('update_user_name'))
my_view.add_url_rule("/update/email/", view_func=UpdateEmail.as_view('update_user_email'))
my_view.add_url_rule("/change/password/", view_func=ChangePassword.as_view('change_password'))

my_view.add_url_rule("/reset/password/", view_func=ResetPassword.as_view('reset_password'))
my_view.add_url_rule("/register/", view_func=Register.as_view('add_users'))
my_view.add_url_rule("/verify/user/", view_func=VerifyUser.as_view('verify_user'))

my_view.add_url_rule("/logout/", view_func=Logout.as_view('logout_user'))

from ChatbotIntegration.views import Chatting, UserChatHistory
my_view.add_url_rule("/user/chatting/", view_func=Chatting.as_view('user_chat'))
my_view.add_url_rule("/user/chat/history/", view_func=UserChatHistory.as_view('user_chat_history'))

from BDI.views import GetQuestions
my_view.add_url_rule("/questions/", view_func=GetQuestions.as_view('get_questions'))

from ChatbotIntegration.views import BDIChat
my_view.add_url_rule("/bdi/chatting/", view_func=BDIChat.as_view('bdi_question_answer'))
