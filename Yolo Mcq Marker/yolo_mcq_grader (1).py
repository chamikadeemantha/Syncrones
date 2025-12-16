from bson import ObjectId
from quart import current_app, jsonify
import base64
import re
import json
import os
from werkzeug.utils import secure_filename
from io import BytesIO
from openai import AsyncOpenAI
from datetime import datetime
from app.constants.error_messages import ErrorMessages
from PIL import Image, ImageOps, ExifTags
import numpy as np
import io
import matplotlib.pyplot as plt
from ultralytics import YOLO
from typing import List, Dict,Tuple
from sklearn.cluster import KMeans



client = AsyncOpenAI

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


def image_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_json_block(text):
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else text.strip()





def preprocess_image(image: Image.Image):
    """
    Auto-orient, grayscale, resize (fit to 1280), pad to 1280x1280 with black.
    Returns the preprocessed PIL image and NumPy array.
    """
    target_size = 1280

    # EXIF orientation fix
    try:
        exif = image._getexif()
        if exif:
            for orientation_tag in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation_tag] == 'Orientation':
                    break
            orientation = exif.get(orientation_tag, 1)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"[WARN] Orientation correction skipped: {e}")

    # Grayscale + Resize + Pad
    image = image.convert("L")
    image.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    image = ImageOps.pad(image, (target_size, target_size), color=0, method=Image.Resampling.LANCZOS)

    return image, np.array(image)

def group_questions_by_columns(question_boxes: List[Dict], image_width: int, num_columns: int = 0) -> List[Dict]:
    """
    Group question boxes into 2 or 3 columns based on x-position, then sort top to bottom within each column.

    Args:
        question_boxes: List of dicts with 'box': [x1, y1, x2, y2]
        image_width: Width of the image
        num_columns: Optional manual override (2 or 3); if 0, it will detect automatically using clustering

    Returns:
        Sorted list of question boxes (dicts) in reading order (top to bottom, left to right)
    """
    # Step 1: Extract X centers
    x_centers = np.array([[(box['box'][0] + box['box'][2]) / 2] for box in question_boxes])

    # Step 2: Use KMeans clustering to detect columns
    if num_columns == 0:
        distortions = []
        for k in [2, 3]:
            km = KMeans(n_clusters=k, n_init="auto").fit(x_centers)
            distortions.append(km.inertia_)
        num_columns = 2 if distortions[0] < distortions[1] * 0.9 else 3  # Prefer simpler model if good enough

    kmeans = KMeans(n_clusters=num_columns, n_init="auto").fit(x_centers)
    labels = kmeans.labels_

    # Step 3: Group by column label
    columns = [[] for _ in range(num_columns)]
    for idx, label in enumerate(labels):
        columns[label].append(question_boxes[idx])

    # Step 4: Sort columns by mean X value (left to right)
    sorted_columns = sorted(columns, key=lambda col: np.mean([(b['box'][0] + b['box'][2]) / 2 for b in col]))

    # Step 5: Sort each column by Y1 (top to bottom)
    for col in sorted_columns:
        col.sort(key=lambda b: b['box'][1])

    # Step 6: Concatenate
    sorted_questions = [q for col in sorted_columns for q in col]
    return sorted_questions

def match_bubbles_to_questions(questions: List[Dict], bubbles: List[Dict]) -> List[Dict]:
    """
    Assign bubbles to their corresponding question box by checking overlap.

    Returns a list of:
    {
        "question_box": {...},
        "answers": [ {"box": ..., "filled": True/False}, ... ]
    }
    """
    grouped = []
    for q in questions:
        qx1, qy1, qx2, qy2 = q['box']
        relevant_bubbles = []
        for b in bubbles:
            bx1, by1, bx2, by2 = b['box']
            # Check if the bubble lies *inside* or near the question box (with small vertical padding)
            if (bx1 >= qx1 and bx2 <= qx2) and (by1 >= qy1 - 10 and by2 <= qy2 + 10):
                relevant_bubbles.append({
                    'box': b['box'],
                    'filled': b['name'] == 'filled'
                })

        # Sort bubbles left-to-right
        relevant_bubbles.sort(key=lambda b: (b['box'][0] + b['box'][2]) / 2)
        grouped.append({'question_box': q, 'answers': relevant_bubbles})

    return grouped

def detect_and_group(image: Image.Image, model, image_width: int = 1280) -> List[Dict]:
    """
    Given a PIL image and a YOLO model, run detection, group questions and bubbles,
    then match bubbles to questions and return grouped data.
    """
    # Assuming you have a preprocess_image function returning processed image and other data
    processed_img, _ = preprocess_image(image)
    results = model(processed_img, conf=0.5)
    
    all_boxes = results[0].boxes.data.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    names = results[0].names
    
    boxes = []
    for i in range(len(all_boxes)):
        cls_name = names[int(class_ids[i])]
        if cls_name in ['question', 'filled', 'unfilled']:
            boxes.append({
                'name': cls_name,
                'box': list(map(int, all_boxes[i][:4]))
            })
    
    question_boxes = [b for b in boxes if b['name'] == 'question']
    bubble_boxes = [b for b in boxes if b['name'] in ['filled', 'unfilled']]
    
    sorted_questions = group_questions_by_columns(question_boxes, image_width=image_width)
    question_with_bubbles = match_bubbles_to_questions(sorted_questions, bubble_boxes)
    
    return question_with_bubbles

def print_grouped_questions(grouped_data: List[Dict]):
    """
    Nicely print the grouped question and bubble data for inspection.
    """
    for idx, qwb in enumerate(grouped_data):
        q_box = qwb['question_box']['box']
        print(f"Question {idx + 1}: Box {q_box}")
        print("Answers:")
        for ans_idx, ans in enumerate(qwb['answers']):
            ans_box = ans['box']
            filled_status = "Filled" if ans['filled'] else "Unfilled"
            print(f"  Answer {ans_idx + 1}: Box {ans_box}, {filled_status}")
        print("-" * 40)

def index_to_letter(index: int) -> str:
    return chr(ord('A') + index)

def compare_answers_by_index(answer_group: List[Dict], mark_group: List[Dict]) -> Dict:
    score = 0
    total_questions = len(mark_group)
    report = {}

    for i, (student_q, correct_q) in enumerate(zip(answer_group, mark_group), start=1):
        student_answers = student_q['answers']
        correct_answers = correct_q['answers']

        # Get indices of filled bubbles
        student_filled_indices = [idx for idx, ans in enumerate(student_answers) if ans['filled']]
        correct_filled_indices = [idx for idx, ans in enumerate(correct_answers) if ans['filled']]

        if len(student_filled_indices) > 1:
            status = "incorrect"
            message = "Multiple answers marked"
        elif student_filled_indices == correct_filled_indices:
            status = "correct"
            message = "Correct"
            score += 1
        else:
            status = "incorrect"
            message = "Incorrect answer"

        # Map indices to letters (e.g. 0 -> 'A', 1 -> 'B', etc)
        student_ans_letters = [index_to_letter(idx) for idx in student_filled_indices]
        correct_ans_letters = [index_to_letter(idx) for idx in correct_filled_indices]

        report[str(i)] = {
            "status": status,
            "message": message,
            "student_answer": student_ans_letters,
            "correct_answer": correct_ans_letters
        }

    percentage = round((score / total_questions) * 100, 2) if total_questions else 0

    return {
        "score": score,
        "total": total_questions,
        "percentage": percentage,
        "report": report
    }

async def grade_answers(files, student_uuid: str, index_number: str, exam_id: str):
    
    # Configuration
    current_app.config['UPLOAD_FOLDER'] = 'uploads'
    current_app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    current_app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload size

    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Check if files were uploaded
    if 'user' not in files or 'key' not in files:
        raise ValueError(f'{ErrorMessages.NO_FILES_PROVIDED}')
    
    student_file = files['user']
    answer_key_file = files['key']
    
    # Check if files are selected
    if student_file.filename == '' or answer_key_file.filename == '':
        raise ValueError(f'{ErrorMessages.NO_FILE_SELECTED}')
    
    if not (allowed_file(student_file.filename) and allowed_file(answer_key_file.filename)):
        raise ValueError(f'{ErrorMessages.INVALID_FILE_TYPE_OR_SIZE}')
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'trained_models', 'best5.pt')#Yolo model name
        model = YOLO(model_path)

        # Save files with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        student_filename = f"student_{timestamp}_{secure_filename(student_file.filename)}"
        answer_key_filename = f"key_{timestamp}_{secure_filename(answer_key_file.filename)}"
        
        student_path = os.path.join(current_app.config['UPLOAD_FOLDER'], student_filename)
        answer_key_path = os.path.join(current_app.config['UPLOAD_FOLDER'], answer_key_filename)
        
        student_file.save(student_path)
        #await student_file.save(student_path)
        answer_key_file.save(answer_key_path)
        
        # Open images
        student_image = Image.open(student_path)
        answer_key_image = Image.open(answer_key_path)
        
        # Convert to base64
        student_base64 = image_to_base64(student_image)
         
        # Process student answer sheet
        response_student = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from images of answer sheets."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a multiple-choice answer sheet. Please extract only the student's handwritten index number, "
                                "usually found at the top or corner of the sheet. Return the result as a JSON object in this format:\n"
                                "{\"student_index\": \"<index_number>\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{student_base64}"}
                        }
                    ]
                }
            ],
            temperature=0
        )
        
      
        # Extract responses
        student_output_text = response_student.choices[0].message.content
          # Parse JSON
        student_json = extract_json_block(student_output_text)
        student_extracted = json.loads(student_json)
        raw_index = student_extracted.get("student_index") or student_extracted.get("index")

        if not raw_index:
            raise ValueError(f'{ErrorMessages.COULD_NOT_DETECT_INDEX_NUMBER}')

        student_index = raw_index if raw_index.lower().startswith("indx") else f"indx{raw_index}"

        # processed_answer_img, _ = preprocess_image(student_image)
        # processed_marks_img, _ = preprocess_image(answer_key_image)
        
        answer_group = detect_and_group(student_image, model)
        mark_group = detect_and_group(answer_key_image, model)

        result = compare_answers_by_index(answer_group, mark_group)

        #print(f"Score: {result['score']}/{result['total']} ({result['percentage']}%)")

        #for q_num, details in result['report'].items():
        #    print(f"Q{q_num}: {details['message']}")
        #    print(f"  Student Answer Boxes: {details['student_answer']}")
        #    print(f"  Correct Answer Boxes: {details['correct_answer']}")
        # Generate result
        return {
            'student_uuid': student_uuid,
            'exam_id': ObjectId(exam_id),
            'index_number': index_number,
            'score': result['score'],
            'total': result['total'],
            'percentage':result['percentage'],
            'report': result['report'],
            'student_image': student_filename,
            'answer_key_image': answer_key_filename,
            'timestamp': datetime.now(),
           
        }
    
    except Exception as e:
        raise ValueError(f'{ErrorMessages.PROCESSING_FAILED}: {str(e)}')