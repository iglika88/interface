#from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
#from flask_bcrypt import Bcrypt
from extensions import bcrypt 

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)



class VocabularyEntry(db.Model):
    __tablename__ = 'vocabulary_entries'
    id = db.Column(db.Integer, primary_key=True)
    item = db.Column(db.String(255), nullable=False)
    pos = db.Column(db.String(50), nullable=True)
    translation = db.Column(db.String(255), nullable=True)
    lesson_title = db.Column(db.String(255), nullable=True)
    reading_or_listening = db.Column(db.String(50), nullable=True)
    course_code = db.Column(db.String(50), nullable=False)
    cefr_level = db.Column(db.String(10), nullable=False)
    domain = db.Column(db.String(255), nullable=False)
    user = db.Column(db.String(50), nullable=False)
    date_loaded = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    number_of_contexts = db.Column(db.Integer, nullable=False, default=0)
    contexts = db.relationship('ContextEntry', backref='vocabulary_entry', lazy=True)


class ContextEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('vocabulary_entries.id'), nullable=False)
    context = db.Column(db.Text, nullable=False)
    target_word = db.Column(db.String(128), nullable=False)
    user = db.Column(db.String(50), nullable=False)
    date_added = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    generation_method = db.Column(db.String(50), nullable=True, default='gemini')



class Exercise(db.Model):
    __tablename__ = 'exercises'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.String(128), nullable=False)
    exercise_type = db.Column(db.String(50), nullable=False)  # e.g., gapfill, mcq
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('exercises', lazy=True))


