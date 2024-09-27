"""
Please use this script to extract context examples for integration into the associated CATS Moodle plugin.

To run the script, please indicate the course code(s) that you would like to extract contexts for,
optionally followed by the minimum number of contexts present per vocabulary item, optionally followed
by the maximum number of contexts present.

Examples:
 python3 extract_info_for_moodle.py LANG1861
   - All contexts for the course LANG1861's vocabulary list, for vocabulary items for which there is a total of 100 contexts, will be extracted.
   
 python3 extract_info_for_moodle.py LANG1861 0
   - All contexts for the course LANG1861's vocabulary list will be extracted.

 python3 extract_info_for_moodle.py LANG1861 30
   - All contexts for the course LANG1861's vocabulary list, for vocabulary items for which there are at least 30 contexts, will be extracted.

 python3 extract_info_for_moodle.py LANG1861 LANGL1171A 30 50

   - All contexts for the course LANG1861's and the course LANGL1171A's vocabulary lists, for vocabulary items for which there are at least 30 but not more than 50 contexts, will be extracted.
"""


import csv
import sys
from app import db, app  # Import the db and app directly from your app.py
from models import VocabularyEntry, ContextEntry

def extract_data_to_csv(course_codes, min_contexts=100, max_contexts=100):
    """
    Extracts vocabulary entries with contexts matching the given course codes and context count range,
    and writes the data to a CSV file.
    """
    # Construct the output filename based on course codes
    course_codes_str = '_'.join(course_codes)
    output_filename = f'vocabulary_contexts_{course_codes_str}.csv'

    # Open a CSV file for writing
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write the header row
        csvwriter.writerow([
            'Item ID', 'Item', 'POS', 'Translation', 'Lesson Title', 'Reading or Listening',
            'Course Code', 'CEFR Level', 'Domain', 'Context', 'Target Word'
        ])

        # Iterate over each course code provided
        for course_code in course_codes:
            # Query to get VocabularyEntry items with context counts within the specified range
            vocabulary_entries = db.session.query(VocabularyEntry).filter(
                VocabularyEntry.number_of_contexts.between(min_contexts, max_contexts),
                VocabularyEntry.course_code == course_code
            ).all()

            # Log the number of entries found for debugging
            print(f"Course Code: {course_code} - Found {len(vocabulary_entries)} entries.")

            # Iterate over each VocabularyEntry and its contexts
            for entry in vocabulary_entries:
                for context in entry.contexts:
                    csvwriter.writerow([
                        entry.id,  # Include the Item ID
                        entry.item,
                        entry.pos,
                        entry.translation,
                        entry.lesson_title,
                        entry.reading_or_listening,
                        entry.course_code,
                        entry.cefr_level,
                        entry.domain,
                        context.context,
                        context.target_word
                    ])
                    
    print(f"Data extraction completed. Output saved to {output_filename}.")

# Ensure the script is run with the correct number of arguments
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_info_for_moodle.py <course_code(s)> [min_contexts] [max_contexts]")
        sys.exit(1)

    # Parse course codes as separate arguments if no comma is used
    course_codes = []
    for arg in sys.argv[1:]:
        if arg.isdigit():
            break
        if ',' in arg:
            course_codes.extend(arg.split(','))
        else:
            course_codes.append(arg)
    
    # Parse min and max contexts
    remaining_args = sys.argv[len(course_codes) + 1:]
    min_contexts = int(remaining_args[0]) if len(remaining_args) > 0 else 100
    max_contexts = int(remaining_args[1]) if len(remaining_args) > 1 else 100

    # Run the extraction within the app context
    with app.app_context():
        extract_data_to_csv(course_codes, min_contexts, max_contexts)
