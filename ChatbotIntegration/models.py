import datetime
from User.db import db
from User.models import User, local_timezone_conversion


class UserChat(db.Document):
    user = db.ReferenceField(User)
    chat_timestamp = db.DateTimeField(default=local_timezone_conversion(datetime.datetime.now()), required=True)
    chat = db.StringField(required=True)
    is_user_chat = db.BooleanField()

    def __str__(self):
        return "{} - {}".format(self.user, self.chat)

