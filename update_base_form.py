# to update the base forms e.g. strip them from 'an', 'to' etc.

from app import app, db, strip_prefix, Vocabulary

# Define the Vocabulary model if not already defined in the imported module
class Vocabulary(db.Model):
    __tablename__ = 'vocabulary'
    id = db.Column(db.Integer, primary_key=True)
    base_form = db.Column(db.String(255), nullable=False)
    # Add other columns here

def update_base_forms():
    with app.app_context():
        # Fetch all entries from the vocabulary table
        vocabulary_entries = Vocabulary.query.all()

        # Iterate over each entry and update the base_form
        for entry in vocabulary_entries:
            original_base_form = entry.base_form
            stripped_base_form = strip_prefix(original_base_form)
            if original_base_form != stripped_base_form:
                entry.base_form = stripped_base_form
                db.session.commit()
                print(f"Updated {original_base_form} to {stripped_base_form}")

if __name__ == "__main__":
    update_base_forms()

