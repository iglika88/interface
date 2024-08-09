from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from models import db, User, VocabularyEntry, ContextEntry, Exercise
from extensions import bcrypt
from nlp_utils import (find_target_word, lemmatize_sentence, check_linguistic_features, clean_text,
                       generate_and_validate_contexts, strip_prefixes, update_pos)

from sqlalchemy.sql import func
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect, CSRFError
import pandas as pd
import os
import re
from datetime import datetime
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sqlalchemy import or_
import logging
import json

from werkzeug.utils import secure_filename
from flask import request
import magic  # libmagic wrapper to check MIME types of uploaded files

from google.api_core.exceptions import ResourceExhausted #note if there is a problem with Gemini quota and use Mistral instead
import openai #for Mistral generation

# in order to export files to Word
from docx import Document
from io import BytesIO

from datetime import timedelta

from functools import wraps #to use decorator to check if user is logged in


nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
csrf = CSRFProtect(app)
app.config['SECRET_KEY'] = 'fj90if0'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1440) #make sure user is logged out automatically after a number of minutes of inactivity; 24 hours currently set 


# Initialize extensions with the Flask app
db.init_app(app)
migrate = Migrate(app, db)
#bcrypt = Bcrypt(app)
bcrypt.init_app(app)
logging.basicConfig(level=logging.DEBUG)


import time
import spacy
import nltk
import google.generativeai as genai
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from flask import jsonify
import langid
import random
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")

#genai.configure(api_key="AIzaSyAFsbEx5fpzsoePD_1Wyct63A8PgBejBKI")  
#model = genai.GenerativeModel('gemini-pro')


# Configuring maximum upload size (e.g., 10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 Megabytes

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
ALLOWED_MIME_TYPES = {'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/vnd.ms-excel', 'text/csv'}


def login_required(f):  #to ensure user is logged in
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_mime_type(file):
    mime = magic.Magic(mime=True)
    file_mime_type = mime.from_buffer(file.read(1024))  # Check the first 1024 bytes
    file.seek(0)  # Reset the file pointer to the start
    return file_mime_type in ALLOWED_MIME_TYPES


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash("Username or email already registered.", 'error')
            return render_template('register.html')

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.permanent = True  # Make the session permanent
            session['username'] = username
            session['user_id'] = user.id
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')



@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('logout_confirmation'))

@app.route('/logout_confirmation')
def logout_confirmation():
    return """
    <p>You have been logged out successfully.</p>
    <p><a href="/">Back to Home Page</a></p>
    """

@app.route('/personal_space/<username>')
def personal_space(username):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))
    return render_template('personal_space.html', username=username)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/add_vocabulary_list', methods=['GET', 'POST'])
@login_required
def add_vocabulary_list():
    if request.method == 'POST':
        course_code = request.form['course_code']
        cefr_level = request.form['cefr_level']
        domain = request.form['domain']
        file = request.files['file']

        if file:
            if not allowed_file(file.filename):
                flash('Invalid file format. Please upload a .csv or .xlsx file.', 'error')
                return redirect(url_for('add_vocabulary_list'))
            if not allowed_mime_type(file):
                flash('Invalid file MIME type. Please upload a file with the correct format.', 'error')
                return redirect(url_for('add_vocabulary_list'))
            
            filename = secure_filename(file.filename)
            file_content = file.read()

            try:
                if filename.endswith('.xlsx'):
                    df = pd.read_excel(file_content)
                elif filename.endswith('.csv'):
                    df = pd.read_csv(file_content.decode('utf-8'))
                else:
                    flash('Invalid file format. Please upload a .csv or .xlsx file.', 'error')
                    return redirect(url_for('add_vocabulary_list'))
            except Exception as e:
                flash(f'Error reading the file: {str(e)}', 'error')
                return redirect(url_for('add_vocabulary_list'))

            required_columns = ['Item', 'POS', 'Translation', 'Lesson title', 'Reading or listening']
            for column in required_columns:
                if column not in df.columns:
                    df[column] = ''

            user = session['username']
            date_loaded = datetime.utcnow()
            number_of_contexts = 0
            contexts = ''

            for index, row in df.iterrows():
                entry = VocabularyEntry(
                    item=strip_prefixes(row['Item']),
                    pos=row['POS'] if row['POS'] else None,
                    translation=row['Translation'],
                    lesson_title=row['Lesson title'],
                    reading_or_listening=row['Reading or listening'],
                    course_code=course_code,
                    cefr_level=cefr_level,
                    domain=domain,
                    user=user,
                    date_loaded=date_loaded,
                    number_of_contexts=number_of_contexts,
                    contexts=contexts
                )
                db.session.add(entry)
            db.session.commit()
            flash('Thank you for uploading a vocabulary list!', 'success')
            return redirect(url_for('home'))
        else:
            flash('No file provided.', 'error')
            return redirect(url_for('add_vocabulary_list'))

    return render_template('add_vocabulary_list.html')

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('The file is too large. Please upload a file smaller than 10MB.', 'error')
    return redirect(url_for('add_vocabulary_list'))


@app.route('/add_context_examples', methods=['GET', 'POST'])
@csrf.exempt  # Disable CSRF protection for this route
@login_required
def add_context_examples():
    # Get unique values for dropdowns and sort them alphabetically
    course_codes = sorted([code.course_code for code in VocabularyEntry.query.with_entities(VocabularyEntry.course_code).distinct().all()])
    domains = sorted([domain.domain for domain in VocabularyEntry.query.with_entities(VocabularyEntry.domain).distinct().all()])
    levels = sorted([level.cefr_level for level in VocabularyEntry.query.with_entities(VocabularyEntry.cefr_level).distinct().all()])
    items = sorted([strip_prefixes(item.item) for item in VocabularyEntry.query.with_entities(VocabularyEntry.item).distinct().all()])

    results = []
    selected_params = {
        'course_code': [],
        'domain': [],
        'cefr_level': [],
        'number_of_contexts': [],
        'item': ''
    }

    if request.method == 'POST':
        search_params = {
            'course_code': request.form.getlist('course_code'),
            'domain': request.form.getlist('domain'),
            'cefr_level': request.form.getlist('cefr_level'),
            'item': request.form.get('item'),  # Get the single selected item
            'number_of_contexts': request.form.getlist('number_of_contexts')
        }
        selected_params = search_params

        query = VocabularyEntry.query
        if search_params['course_code']:
            query = query.filter(VocabularyEntry.course_code.in_(search_params['course_code']))
        if search_params['domain']:
            query = query.filter(VocabularyEntry.domain.in_(search_params['domain']))
        if search_params['cefr_level']:
            query = query.filter(VocabularyEntry.cefr_level.in_(search_params['cefr_level']))
        if search_params['item']:
            query = query.filter(VocabularyEntry.item == search_params['item'])
        if search_params['number_of_contexts']:
            context_filters = []
            for context_range in search_params['number_of_contexts']:
                if context_range == 'less_than_10':
                    context_filters.append(VocabularyEntry.number_of_contexts < 10)
                elif context_range == '10_to_20':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(10, 20))
                elif context_range == '20_to_50':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(20, 50))
                elif context_range == '50_to_100':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(50, 100))
            query = query.filter(or_(*context_filters))

        results = query.order_by(VocabularyEntry.item).all()  # Sort results alphabetically by item

        # Add actual count of context examples to each result
        for result in results:
            context_count = ContextEntry.query.filter_by(item_id=result.id).count()
            result.number_of_contexts = context_count

    return render_template('add_context_examples.html',
                           course_codes=course_codes,
                           domains=domains,
                           levels=levels,
                           items=items,
                           results=results,
                           selected_params=selected_params)



@app.route('/current_contexts/<int:item_id>', methods=['GET'])
@login_required
def current_contexts(item_id):
    item = VocabularyEntry.query.get_or_404(item_id)
    contexts = ContextEntry.query.filter_by(item_id=item_id).all()
    return render_template('current_contexts.html', item=item, contexts=contexts)



@app.route('/generate_context/<int:item_id>', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def generate_context(item_id):
    item = VocabularyEntry.query.get_or_404(item_id)
    generated_contexts = None

    if request.method == 'POST':
        if 'generate_contexts' in request.form:
            pos = item.pos
            domain = item.domain
            level = item.cefr_level
            generated_contexts = generate_and_validate_contexts(item.item, pos, domain, level)

        elif 'save_contexts' in request.form:
            selected_contexts = request.form.getlist('selected_contexts')
            context_texts = request.form.getlist('context_texts')
            user = session.get('username')
            date_added = datetime.utcnow()
            generation_methods = request.form.getlist('generation_methods')

            for idx in selected_contexts:
                idx = int(idx) - 1
                if idx < len(context_texts):
                    context = context_texts[idx]
                    target_word = find_target_word(context, item.item)
                    generation_method = generation_methods[idx]
                    new_context = ContextEntry(
                        item_id=item_id,
                        context=context,
                        target_word=target_word,
                        user=user,
                        date_added=date_added,
                        generation_method=generation_method
                    )   
                    db.session.add(new_context)

            # Update the number_of_contexts field in VocabularyEntry
            item.number_of_contexts = ContextEntry.query.filter_by(item_id=item_id).count()
            db.session.commit()
            flash('Selected contexts saved successfully.', 'success')


    # Get current contexts with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = ContextEntry.query.filter_by(item_id=item_id).paginate(page=page, per_page=per_page, error_out=False)
    current_contexts = pagination.items

    # Find and highlight target words
    highlighted_contexts = []
    for context in current_contexts:
        target_word = find_target_word(context.context, item.item)
        highlighted_context = re.sub(f"\\b{target_word}\\b", f"<b>{target_word}</b>", context.context, flags=re.IGNORECASE)
        highlighted_contexts.append(highlighted_context)

    return render_template('generate_context.html', item=item, current_contexts=highlighted_contexts, generated_contexts=generated_contexts, pagination=pagination, per_page=per_page, page=page, find_target_word=find_target_word)



@app.route('/create_exercises', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def create_exercises():
    # Get unique values for checkboxes and sort them alphabetically
    course_codes = sorted([code.course_code for code in VocabularyEntry.query.with_entities(VocabularyEntry.course_code).distinct().all()])
    domains = sorted([domain.domain for domain in VocabularyEntry.query.with_entities(VocabularyEntry.domain).distinct().all()])
    levels = sorted([level.cefr_level for level in VocabularyEntry.query.with_entities(VocabularyEntry.cefr_level).distinct().all()])
    context_ranges = ['less_than_10', '10_to_20', '20_to_50', '50_to_100']

    items = sorted([strip_prefixes(item.item) for item in VocabularyEntry.query.with_entities(VocabularyEntry.item).distinct().all()])

    results = []
    selected_params = {}

    if request.method == 'POST':
        search_params = {
            'course_code': request.form.getlist('course_code'),
            'domain': request.form.getlist('domain'),
            'cefr_level': request.form.getlist('cefr_level'),
            'item': request.form.get('item'),
            'number_of_contexts': request.form.getlist('number_of_contexts')
        }
        selected_params = search_params

        query = VocabularyEntry.query
        if search_params['course_code']:
            query = query.filter(VocabularyEntry.course_code.in_(search_params['course_code']))
        if search_params['domain']:
            query = query.filter(VocabularyEntry.domain.in_(search_params['domain']))
        if search_params['cefr_level']:
            query = query.filter(VocabularyEntry.cefr_level.in_(search_params['cefr_level']))
        if search_params['item']:
            query = query.filter(VocabularyEntry.item == search_params['item'])
        if search_params['number_of_contexts']:
            context_filters = []
            for context_range in search_params['number_of_contexts']:
                if context_range == 'less_than_10':
                    context_filters.append(VocabularyEntry.number_of_contexts < 10)
                elif context_range == '10_to_20':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(10, 20))
                elif context_range == '20_to_50':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(20, 50))
                elif context_range == '50_to_100':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(50, 100))
            query = query.filter(or_(*context_filters))
        
        results = query.order_by(VocabularyEntry.item).all()  # Sort results alphabetically by item

        # Debugging output to verify the results
        print("Search Results:")
        for result in results:
            print(f"Item: {result.item}, Number of Contexts: {result.number_of_contexts}, ID: {result.id}")

    return render_template('create_exercises.html',
                           course_codes=course_codes,
                           domains=domains,
                           levels=levels,
                           context_ranges=context_ranges,
                           items=items,
                           results=results,
                           selected_params=selected_params)




@app.route('/filter_items', methods=['POST'])
def filter_items():
    try:
        data = request.get_json()
        logging.debug(f"Received data: {data}")
        course_codes = data.get('course_code', [])
        domains = data.get('domain', [])
        cefr_levels = data.get('cefr_level', [])
        context_ranges = data.get('number_of_contexts', [])

        query = VocabularyEntry.query

        if course_codes:
            query = query.filter(VocabularyEntry.course_code.in_(course_codes))
        if domains:
            query = query.filter(VocabularyEntry.domain.in_(domains))
        if cefr_levels:
            query = query.filter(VocabularyEntry.cefr_level.in_(cefr_levels))
        if context_ranges:
            context_filters = []
            for context_range in context_ranges:
                if context_range == 'less_than_10':
                    context_filters.append(VocabularyEntry.number_of_contexts < 10)
                elif context_range == '10_to_20':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(10, 20))
                elif context_range == '20_to_50':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(20, 50))
                elif context_range == '50_to_100':
                    context_filters.append(VocabularyEntry.number_of_contexts.between(50, 100))
            query = query.filter(or_(*context_filters))

        items = sorted([strip_prefixes(item.item) for item in query.with_entities(VocabularyEntry.item).distinct().all()])

        logging.debug(f"Filtered items: {items}")
        return jsonify(items)
    except Exception as e:
        logging.error(f"Error in filter_items: {e}")
        return jsonify({'error': str(e)}), 400



@app.route('/create_gapfill/<int:context_id>', methods=['GET', 'POST'])
@csrf.exempt  # Disable CSRF protection for this route
@login_required
def create_gapfill(context_id):
    context = ContextEntry.query.get_or_404(context_id)
    item = VocabularyEntry.query.get_or_404(context.item_id)
    
    target_word = context.target_word
    translation = item.translation
    gapfill_question = context.context.replace(target_word, f"{target_word[0]}{' _' * (len(target_word) - 1)} ({translation})")

    if request.method == 'POST':
        question = request.form['question']
        answer = request.form['answer']
        
        # Save the exercise to the user's list
        exercise = Exercise(
            user_id=session['user_id'],
            question=question,
            answer=answer,
            exercise_type='gapfill'
        )
        db.session.add(exercise)
        db.session.commit()

        flash('Exercise saved successfully!', 'success')
        return redirect(url_for('personal_space', username=session['username']))

    return render_template('create_gapfill.html', question=gapfill_question, answer=target_word)



@app.route('/create_mcq/<int:context_id>', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def create_mcq(context_id):
    context = ContextEntry.query.get_or_404(context_id)
    item = VocabularyEntry.query.get_or_404(context.item_id)

    def get_distractors(item, exclude_list, num_distractors=4):
        distractors_query = VocabularyEntry.query.filter(
            VocabularyEntry.course_code == item.course_code,
            VocabularyEntry.pos == item.pos,
            VocabularyEntry.id != item.id,
            ~VocabularyEntry.item.in_(exclude_list)
        ).all()
        
        distractors = [d.item for d in distractors_query]
        if len(distractors) < num_distractors:
            distractors += ['-------'] * (num_distractors - len(distractors))
        
        return random.sample(distractors, min(num_distractors, len(distractors)))

    def generate_mcq(context, item):
        distractors = get_distractors(item, [])
        all_options = [item.item] + distractors
        random.shuffle(all_options)
        question = context.context.replace(context.target_word, '__________')
        return question, all_options

    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            if data.get('action') == 'change_distractors':
                selected_distractors = data.get('selected_distractors', [])
                current_options = data.get('current_options', [])

                new_distractors = get_distractors(item, [item.item] + [opt for opt in current_options if opt not in selected_distractors], len(selected_distractors))

                updated_options = [opt if opt not in selected_distractors else new_distractors.pop(0) for opt in current_options]

                return jsonify({'new_options': updated_options})

        question = request.form['question']
        answer = request.form['answer']
        options = request.form.getlist('options')
        options_str = '\n'.join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])

        exercise = Exercise(
            user_id=session['user_id'],
            question=f"{question}\n{options_str}",
            answer=answer,
            exercise_type='mcq'
        )
        db.session.add(exercise)
        db.session.commit()

        flash(f"Exercise saved successfully! {session['username']}'s Personal Page", 'success')
        return redirect(url_for('personal_space', username=session['username']))

    question, options = generate_mcq(context, item)
    answer = context.target_word

    return render_template('create_mcq.html', question=question, options=options, answer=answer, context_id=context_id)


@app.template_filter('to_letter')
def to_letter(value):
    return chr(value + 64)


@app.template_filter('plus_64')
def plus_64(value):
    return chr(value + 64)


@app.route('/generate_exercises/<int:item_id>', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def generate_exercises(item_id):
    item = VocabularyEntry.query.get_or_404(item_id)

    # Get current contexts with pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = ContextEntry.query.filter_by(item_id=item_id).paginate(page=page, per_page=per_page, error_out=False)
    current_contexts = pagination.items

    # Find and highlight target words
    highlighted_contexts = []
    for context in current_contexts:
        target_word = find_target_word(context.context, item.item)
        highlighted_context = re.sub(f"\\b{target_word}\\b", f"<b>{target_word}</b>", context.context, flags=re.IGNORECASE)
        # Create a new object to store the highlighted context and the context ID
        highlighted_context_obj = {
            'id': context.id,
            'context': highlighted_context
        }
        highlighted_contexts.append(highlighted_context_obj)

    return render_template('generate_exercises.html', item=item, current_contexts=highlighted_contexts, pagination=pagination, per_page=per_page, page=page)


@app.route('/my_exercises/<username>')
@login_required
def my_exercises(username):
    if 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    user = User.query.filter_by(username=username).first_or_404()
    exercises = Exercise.query.filter_by(user_id=user.id).all()

    for exercise in exercises:
        exercise.question = format_options(exercise.question)

    return render_template('my_exercises.html', exercises=exercises)


@app.route('/edit_exercise', methods=['POST'])
@csrf.exempt
@login_required
def edit_exercise():
    data = request.get_json()
    exercise_id = data.get('exercise_id')
    field = data.get('field')
    new_value = data.get('new_value')

    exercise = Exercise.query.get_or_404(exercise_id)
    if field == 'question':
        exercise.question = new_value
    elif field == 'answer':
        exercise.answer = new_value
    db.session.commit()

    return jsonify({'success': True})

@app.route('/delete_exercises', methods=['POST'])
@csrf.exempt
@login_required
def delete_exercises():
    selected_exercises = request.form.getlist('selected_exercises')
    for exercise_id in selected_exercises:
        exercise = Exercise.query.get_or_404(exercise_id)
        db.session.delete(exercise)
    db.session.commit()

    flash('Selected exercises deleted successfully!', 'success')
    return redirect(url_for('my_exercises', username=session['username']))


@app.route('/export_to_word', methods=['GET', 'POST'])
@csrf.exempt
@login_required
def export_to_word():
    include_answers = request.form.get('include_answers') == 'true'
    user_id = session['user_id']
    user = User.query.get_or_404(user_id)
    exercises = Exercise.query.filter_by(user_id=user.id).all()

    doc = Document()
    doc.add_heading(f"{user.username}'s Exercises", 0)

    for idx, exercise in enumerate(exercises, 1):
        doc.add_paragraph(f"{idx}. {exercise.question}")
        if include_answers:
            doc.add_paragraph(f"Answer: {exercise.answer}")
        #doc.add_paragraph("\n")

    if not os.path.exists('exports'):
        os.makedirs('exports')

    file_path = f'exports/{user.username}_exercises.docx'
    doc.save(file_path)

    return send_file(file_path, as_attachment=True)



@app.template_filter('escapejs')
def escapejs_filter(s):
    if s is None:
        return ''
    return json.dumps(s)[1:-1]


def format_options(question):
    # Adjust the pattern to handle line breaks and spaces appropriately
    pattern = r'(A\..*?)(?:\s*)(B\..*?)(?:\s*)(C\..*?)(?:\s*)(D\..*?)(?:\s*)(E\..*?)'
    match = re.search(pattern, question, re.DOTALL)
    if match:
        print("Match found:", match.groups())  # Debugging output
        formatted_options = "\t".join(match.groups())  # Join options with a tab
        formatted_question = re.sub(pattern, formatted_options, question, flags=re.DOTALL)
        print("Formatted question:", formatted_question)  # Debugging output
        return formatted_question
    print("No match found for question:", question)  # Debugging output
    return question



if __name__ == '__main__':
    from nlp_utils import update_pos
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    #update_pos(app)  # UNCOMMENT TO AUTOMATICALLY UPDATE ALL POS IN THE DATABASE
    app.run(debug=True)

