import datetime
from mongoengine import Document, StringField, FloatField, IntField, ListField, DictField, DateTimeField

class Analysis(Document):
    original_image = StringField(required=True)
    processed_image = StringField(required=True)
    vascular_bundle_image = StringField(required=True)
    total_root_image = StringField(required=True)
    xylem_image = StringField(required=True)

    vascular_area = FloatField(required=True)
    vascular_diameter = FloatField(required=True)
    xylem_count = IntField(required=True)
    xylem_diameter = FloatField(required=True)
    xylem_details = ListField(DictField())

    created_at = DateTimeField(default=datetime.datetime.utcnow)