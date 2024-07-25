import os
from app_upd import app, db
from app_upd import VocabularyEntry  # Adjust the import to match your model

# Define the function
def strip_prefixes(item):
    prefixes = ['to ', 'a ', 'an ', 'the ']
    for prefix in prefixes:
        if item.lower().startswith(prefix):
            return item[len(prefix):]
    return item

# Create a function to update all entries
def update_items():
    with app.app_context():
        items = VocabularyEntry.query.all()
        for item in items:
            original_item = item.item
            stripped_item = strip_prefixes(original_item)
            if original_item != stripped_item:
                item.item = stripped_item
                db.session.add(item)
        db.session.commit()
        print("Database update complete")

if __name__ == "__main__":
    update_items()

