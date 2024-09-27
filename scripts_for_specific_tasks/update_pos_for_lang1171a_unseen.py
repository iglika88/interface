import sys
import os
from nltk import pos_tag
from app import app, db
from models import VocabularyEntry
from nlp_utils import get_wordnet_pos  # Ensure this function exists in your nlp_utils.py

def update_pos_for_specific_course_code(app, course_code):
    with app.app_context():
        # Query all entries with the specific course code
        entries = VocabularyEntry.query.filter_by(course_code=course_code).all()
        for entry in entries:
            if not entry.pos or entry.pos == 'word':
                if ' ' in entry.item:
                    entry.pos = 'expression'
                else:
                    nltk_tag = pos_tag([entry.item])[0][1]
                    entry.pos = get_wordnet_pos(nltk_tag)
        db.session.commit()

# Run the function for the specific course code
if __name__ == "__main__":
    update_pos_for_specific_course_code(app, 'LANGL1171A_unseen')

