import datetime
from User.db import db
from User.models import User, local_timezone_conversion


class BDI(db.Document):
    response = db.ListField()
    question_no = db.IntField()

    def __str__(self):
        return "{} - {}".format(self.question_no, self.response)

