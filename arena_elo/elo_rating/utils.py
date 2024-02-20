from datetime import datetime
import pytz
import PIL
import os 

def detect_language(text: str) -> str:
    """Detect the langauge of a string."""
    import polyglot  # pip3 install polyglot pyicu pycld2
    from polyglot.detect import Detector
    from polyglot.detect.base import logger as polyglot_logger
    import pycld2

    polyglot_logger.setLevel("ERROR")

    try:
        lang_code = Detector(text).language.name
    except (pycld2.error, polyglot.detect.base.UnknownLanguage):
        lang_code = "unknown"
    return lang_code


def get_time_stamp_from_date(date_str:str):
    """
    Convert a date string to a Unix timestamp
    Args:
        date_str (str): The input date string in the format 'YYYY-MM-DD-HH:MM-TZ', e.g. '2024-02-10-14:00-PT'
    """
    
    # Convert the date string into a format that Python's datetime can understand
    # and specify the correct timezone for PT, which is 'US/Pacific'
    date_format = "%Y-%m-%d-%H:%M-%Z"

    # Parse the date string into a datetime object
    # Note: PT is not directly recognized by pytz, so we manually map it to 'US/Pacific'
    timezone_map = {
        "PT": "US/Pacific",
    }

    # Extract the timezone abbreviation
    tz_abbr = date_str.split("-")[-1]
    # Map the abbreviation to a pytz timezone
    tz_info = pytz.timezone(timezone_map[tz_abbr])

    # Remove the timezone abbreviation for parsing
    date_str_parsed = date_str.rsplit("-", 1)[0]

    # Create a datetime object with the corresponding timezone
    dt = datetime.strptime(date_str_parsed, "%Y-%m-%d-%H:%M").replace(tzinfo=tz_info)

    # Convert the datetime object to a Unix timestamp
    unix_timestamp = dt.timestamp()
    return unix_timestamp

def get_date_from_time_stamp(unix_timestamp: int):
    # Create a datetime object from the Unix timestamp
    dt = datetime.fromtimestamp(unix_timestamp)

    # Convert the datetime object to a string with the desired format
    date_str = dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return date_str


def get_input_image_path(tstamp, conv_id):
    # from tstamp to date e.g. 2024-02-10
    date_str = datetime.fromtimestamp(tstamp, tz=pytz.timezone("US/Pacific")).strftime("%Y-%m-%d")
    LOGDIR = os.getenv("LOGDIR")
    return f"{LOGDIR}/{date_str}-convinput_images/input_image_{conv_id}.png"

def load_image_from_path(image_path):
    # Load the image from the specified
    # path using the Python Imaging Library (PIL)
    try:
        image = PIL.Image.open(image_path)
        return image
    except FileNotFoundError:
        print(f"Image not found at path: {image_path}")
        return None
    except PIL.UnidentifiedImageError:
        print(f"Unidentified image format at path: {image_path}")
        return None
    

    