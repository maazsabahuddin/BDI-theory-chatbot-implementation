
TWILIO_ACCOUNT_SID_old = 'AC94a5281e5600b2303e1ed17adb02b2a7'
TWILIO_AUTH_TOKEN_old = 'a6ca16a2a6d109987a7379cec069c04b'

TWILIO_ACCOUNT_SID = 'AC9a2effc3dd3dcc903cf34863711b5cbc'
TWILIO_AUTH_TOKEN = 'e4bb7958de69119134c100b72c78dd81'

# Changes as per business requirements.
OTP_COUNTER_LIMIT = 5

# kitni der tk otp valid rhega.
OTP_VALID_TIME = 1200

# Initial Counter
OTP_INITIAL_COUNTER = 1

# Country code
COUNTRY_CODE_PK = "92"

# Regex Phone Number
PHONE_NUMBER_REGEX = r"\+" + COUNTRY_CODE_PK + r"3\d{9}"

# Email REGEX
EMAIL_REGEX = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"

# Email Verification
EMAIL_VERIFICATION = False

# Phone Verification
PHONE_VERIFICATION = True
PRIORITY_QUEUE = ["Twilio", "Other"]


SENDER_PHONE_NUMBER_old = '+12029533093'
SENDER_PHONE_NUMBER = '+19388882997'

# Not catchable error message
NOT_CATCHABLE_ERROR_MESSAGE = "Connection problem, please try again."
NOT_CATCHABLE_ERROR_CODE = 400

EMAIL_HOST_USER = ''
