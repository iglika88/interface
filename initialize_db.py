from app import app, db, VocabularyEntry
import pandas as pd
from datetime import datetime

# Load initial data
df = pd.read_excel('voc_list.xlsx')

required_columns = ['Item', 'POS', 'Translation', 'Lesson title', 'Reading or listening', 'Course code', 'CEFR level', 'Domain']
for column in required_columns:
    if column not in df.columns:
        df[column] = ''

user = 'admin'  # Your current username
date_loaded = datetime.utcnow()
number_of_contexts = 0

# Ensure the code runs within the application context
with app.app_context():
    for index, row in df.iterrows():
        entry = VocabularyEntry(
            item=row['Item'],
            pos=row['POS'],
            translation=row['Translation'],
            lesson_title=row['Lesson title'],
            reading_or_listening=row['Reading or listening'],
            course_code=row['Course code'],
            cefr_level=row['CEFR level'],
            domain=row['Domain'],
            user=user,
            date_loaded=date_loaded,
            number_of_contexts=number_of_contexts
        )
        db.session.add(entry)
    db.session.commit()

