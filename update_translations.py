from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Same as in your app.py
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define your model (example model)
class Vocabulary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    base_form = db.Column(db.String(100))
    Traduction = db.Column(db.String(100))
    POS = db.Column(db.String(50))

def update_traductions():
    # Query all records
    records = Vocabulary.query.all()
    for record in records:
        # Strip leading/trailing spaces and lowercase the first letter
        modified_traduction = record.Traduction.strip()
        if modified_traduction:
            modified_traduction = modified_traduction[0].lower() + modified_traduction[1:]
        
        # Update the record
        record.Traduction = modified_traduction
        db.session.add(record)
    
    # Commit the changes to the database
    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        update_traductions()

