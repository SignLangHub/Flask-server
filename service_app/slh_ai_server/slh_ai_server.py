# SingLangHub AI Server code
from flask import Flask, request, jsonify, render_template
import mysql.connector
import json
import openai
import os
import re
# 수어 사전 검색을 위한 라이브러리
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from jamo import h2j, j2h, is_jamo, InvalidJamoError
from collections import Counter
import tempfile  # 추가된 임포트

# HTML 파일 경로 설정
app = Flask(__name__, template_folder='templates', static_folder='static')  # 템플릿 폴더 명시적 설정

# OpenAI API 키 설정
openai.api_key = "{GPT API key}"

# 지침 기본 텍스트
guidelines = (
    "이 지피티는 한국어 문장을 입력 받으면 한국 수어로 문법을 변환해주는 지피티입니다. "
    "규칙은 다음을 따릅니다: "
    "1. 불필요한 조사 및 연결어 제거, "
    "2. 지역명 단순화 및 나열, "
    "3. 구체적인 설명 단어 사용, "
    "4. 불필요한 명사나 형용사 생략, "
    "5. 동사 및 행위 중심 표현, "
    "6. 시각적이고 간결한 표현 사용, "
    "7. 명령형 표현, "
    "8. 원인과 결과 표현 간결화, "
    "9. 의미 강조를 위한 반복, "
    "10. 주어와 목적어 날짜는 문장의 맨 앞으로, "
    "11. 6하 원칙은 문장의 맨 뒤로, "
    "12. 지시 표현의 사용.\n\n"
    "문법 변환의 예시는 아래와 같습니다:\n"
)
# JSON 파일에서 kor와 ksl 필드를 읽고 예시로 추가
json_path = os.path.join(app.root_path, 'templates', 'tuning_data', 'output_data_v1.0.0.json')
ins_limit = 5 # => 지침 예시 데이터 제한

# 모델 경로
model_path = os.path.join(app.root_path, 'templates', 'slh_sl_models', 'gesture_recognizer.task')

with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
    example_texts = ""
    for i, item in enumerate(data, 1):
        example_texts += f"{i}. kor: \"{item['kor']}\" -> ksl: \"{item['ksl']}\"\n"
        # 원하는 만큼의 예시를 추가하기 위해 break 조건을 설정할 수도 있습니다.
        if i >= ins_limit:  # 예시 5개로 제한 (필요 시 수정)
            break
     
     
# MySQL 데이터베이스 설정
db_config = {
    'host': '',                 # 데이터베이스 호스트
    'port':1111,                # 포트
    'user': '',                 # 데이터베이스 사용자 이름
    'password': '',             # 데이터베이스 비밀번호
    'database': 'SLH'           # 데이터베이스 이름
}   
# 지침에 예시 추가
guidelines += example_texts
print(guidelines)
@app.route('/')
def index():
    print('index page')
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_input = data.get('text')

    if not user_input:
        return jsonify({'error': 'No input text provided'}), 400

    try:
        # OpenAI ChatCompletion API 호출 (messages 파라미터 사용)
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 또는 본인의 커스텀 GPT 모델 이름/ID 사용
            messages=[
                {"role": "system", "content": guidelines},
                {"role": "user", "content": user_input}
            ],
            max_tokens=150
        )
        response_text = response['choices'][0]['message']['content'].strip()
        return jsonify({'response': response_text})
    except Exception as e:
        # 예외 처리 및 오류 로그 출력
        print(f"Error: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

# gpt end point
def fetch_related_terms_from_gpt(prompt):
    print(prompt)
    # Adjusted instruction guide with more explicit direction
    instuction_guide = (
        "입력된 {prompt}와 관련된 단어와 특징을 하나의 하나의 [{prompt}, '', '', ...] 이런식의 JSON 배열로 응답해주면 됩니다. "
        "응답은 하나의 JSON 배열 형식으로만 작성하며, 입력된 단어와 관련된 단어를 포함해야 합니다. "
        "지시사항: "
        "1. 단어 끝에 ‘하다’, ‘다’, ‘~함’ 등의 접미사는 제거하고, 원형 단어를 사용합니다. "
        "2. 입력된 단어와 연관성이 높은 단어와 특징을 포함합니다. 예를들어 먹는 떡이면 쌀, 끈적끈적 등 재료와 성질을 포함합니다. "
        "3. 응답은 입력된 키워드와 직접적으로 연관된 내용만 포함해야 하며, 중복 단어는 제거합니다."
        "4. 배열외에 설명이나 키 값 등 부수적인 정보는 포함하지 않습니다."
    )
    try:
        # Call the GPT API with the instructions and prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instuction_guide},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100  # Adjusted to allow enough space for the response
        )

        # Extract the text response and clean it
        response_text = response.choices[0].message['content'].strip()

        # Use a regular expression to extract only the JSON array part
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_array_str = json_match.group(0)
            try:
                parsed_json = json.loads(json_array_str)
                print(parsed_json)
                return parsed_json
            except json.JSONDecodeError as json_err:
                print(f"JSON parsing error: {json_err}")
                print("Raw response for debugging:", response_text)
                return []
        else:
            print("The response is not in the correct JSON format.")
            print("Raw response for debugging:", response_text)
            return []

    except Exception as e:
        print(f"Error fetching data from GPT: {e}")
        return []


# API 엔드포인트
@app.route('/search', methods=['GET'])
def search_keyword():
    keyword = request.args.get('keyword')
    return search_keyword_in_db(keyword, db_config, fetch_related_terms_from_gpt)

# 텍스트 검색 테스트 api
@app.route('/search-test', methods=['GET'])
def search_keyword_test():
    keyword = request.args.get('keyword')
    if not keyword:
        #return jsonify({'error': 'No keyword provided'}), 400
        return jsonify([]), 200 
    try:
        # MySQL 연결 설정
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # SQL 쿼리 작성 및 실행
        query = """
        SELECT d.id, 
               CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '"', ''), '\\\\', '')) AS keyword,
               s.video_url
        FROM SLH.tn_def AS d
        JOIN SLH.tn_sign AS s ON d.id = s.id
        WHERE CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '"', ''), '\\\\', '')) LIKE %s;
        """
        cursor.execute(query, (f'%{keyword}%',))
        results = cursor.fetchall()

        # 데이터베이스 연결 및 커서 닫기
        cursor.close()
        conn.close()

        if not results:
            return jsonify([]), 200 

        return jsonify(results), 200

    except mysql.connector.Error as err:
        return jsonify({'error': f'Database error: {str(err)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    
# 영상으로 검색
@app.route('/search/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    video_file = request.files['file']
    
    if video_file.filename == '':
        return "No selected file", 400
    
    keyword = video_classification(video_file, model_path)
    
    return search_keyword_in_db(keyword, db_config, fetch_related_terms_from_gpt)

@app.route('/search-test/upload', methods=['POST'])
def upload_file_test():
    if 'file' not in request.files:
        return "No file part", 400
    video_file = request.files['file']
    
    if video_file.filename == '':
        return "No selected file", 400
    
    keyword = video_classification(video_file, model_path)
    
    return search_keyword_in_db(keyword, db_config, fetch_related_terms_from_gpt)

# 데이터베이스 검색
def search_keyword_in_db(keyword, db_config, fetch_related_terms_from_gpt):
    """
    주어진 키워드를 기반으로 데이터베이스를 검색하고, 결과가 없으면 GPT 서버를 사용하여 관련 단어로 재검색.

    :param keyword: 검색할 키워드
    :param db_config: MySQL 연결 설정 사전
    :param fetch_related_terms_from_gpt: GPT 서버에서 관련 단어를 가져오는 함수
    :return: 검색 결과 JSON 응답 및 상태 코드
    """
    if not keyword:
        #return jsonify({'error': 'No keyword provided'}), 400
        return jsonify([]), 200 
    try:
        # MySQL 연결 설정
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # SQL 쿼리 작성 및 실행
        query = """
        SELECT d.id, 
               CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '"', ''), '\\\\', '')) AS keyword,
               s.video_url
        FROM SLH.tn_def AS d
        JOIN SLH.tn_sign AS s ON d.id = s.id
        WHERE CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '"', ''), '\\\\', '')) LIKE %s;
        """
        cursor.execute(query, (f'%{keyword}%',))
        results = cursor.fetchall()

        if not results:
            # GPT API를 사용하여 관련 단어 가져오기
            related_terms = fetch_related_terms_from_gpt(keyword)

            if related_terms:
                # 관련 단어들을 사용하여 SQL 쿼리 재작성
                like_conditions = " OR ".join([
                    "CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '\"', ''), '\\\\', '')) LIKE %s"
                    for _ in related_terms
                ])
                query = f"""
                SELECT d.id, 
                       CONCAT(d.ref_word, REPLACE(REPLACE(d.def, '"', ''), '\\\\', '')) AS keyword,
                       s.video_url
                FROM SLH.tn_def AS d
                JOIN SLH.tn_sign AS s ON d.id = s.id
                WHERE {like_conditions};
                """
                params = [f"%{term}%" for term in related_terms]
                cursor.execute(query, params)
                results = cursor.fetchall()

        # 데이터베이스 연결 닫기
        cursor.close()
        conn.close()

        if not results:
            return jsonify([]), 200 

        return jsonify(results), 200

    except mysql.connector.Error as err:
        return jsonify({'error': str(err)}), 500

# 비디오 파일 분류(현재는 손 제스처 모션 캡쳐로 진행 하였음)import cv2
def video_classification(video_data, model_path, threshold=3):
    # 기본 옵션 설정
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = vision.GestureRecognizer
    GestureRecognizerOptions = vision.GestureRecognizerOptions
    VisionRunningMode = vision.RunningMode

    # GestureRecognizer 옵션 설정
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )

    # 인식된 제스처 결과를 저장할 리스트
    recognized_gestures = []

    # video_data가 FileStorage 객체일 경우 read()로 데이터를 읽어옴
    video_bytes = video_data.read() if hasattr(video_data, 'read') else video_data

    # 임시 파일로 비디오 데이터를 저장
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video_file:
        temp_video_file.write(video_bytes)
        temp_video_path = temp_video_file.name

    try:
        # 제스처 인식기 생성
        with GestureRecognizer.create_from_options(options) as recognizer:
            # 임시 파일에서 영상 파일 불러오기
            cap = cv2.VideoCapture(temp_video_path)
            frame_index = 0  # 프레임 인덱스 초기화

            if not cap.isOpened():
                print("Error: 비디오를 불러올 수 없습니다.")
                return []

            print("비디오를 성공적으로 불러왔습니다.")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("End of video or cannot read the frame.")
                    break

                # BGR 이미지를 RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe 이미지 객체 생성
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # 제스처 인식 수행 (VIDEO 모드)
                gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_index)

                # 인식 결과 확인 및 리스트에 추가
                if gesture_recognition_result and gesture_recognition_result.gestures:
                    for gesture_list in gesture_recognition_result.gestures:
                        if gesture_list and hasattr(gesture_list, '__iter__'):  # 리스트인지 확인
                            gesture = gesture_list[0]  # 첫 번째 요소 사용
                            if gesture.category_name.strip():  # 공백 확인
                                #print(f"Frame {frame_index}: Gesture - {gesture}")
                                recognized_gestures.append(gesture.category_name)
                            #else:
                                #print(f"Frame {frame_index}: Skipped empty or invalid gesture")
                #else:
                    #print(f"Frame {frame_index}: No gesture recognized.")

                # 프레임에 결과 표시 (선택 사항)
                if recognized_gestures:
                    cv2.putText(frame, recognized_gestures[-1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                frame_index += 1  # 프레임 인덱스 증가

                # 'q' 키를 누르면 종료
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    # 후처리 기능 적용: 중복 제거 및 공백 제거
    return post_process_gestures(recognized_gestures, threshold)

# 후처리 기능: 중복 제거 및 공백 제거 및 평균 빈도수 필터링
def post_process_gestures(gestures, threshold=3):
    if not gestures:
        return []

    # 각 제스처의 빈도수 계산
    gesture_counts = Counter(gestures)
    total_gestures = sum(gesture_counts.values())
    average_frequency = total_gestures / len(gesture_counts)

    print(f"\nGesture counts: {gesture_counts}")
    print(f"Average frequency: {average_frequency:.2f}")

    # 평균 빈도수보다 적은 제스처는 제외
    filtered_gestures = [gesture for gesture in gestures if gesture_counts[gesture] >= average_frequency]

    processed_gestures = []
    count = 1

    for i in range(1, len(filtered_gestures)):
        if filtered_gestures[i] == filtered_gestures[i - 1]:
            count += 1
        else:
            if count > threshold:
                processed_gestures.append(filtered_gestures[i - 1])
            else:
                processed_gestures.extend(filtered_gestures[i - count:i])
            count = 1

    # 마지막 요소 처리
    if count > threshold:
        processed_gestures.append(filtered_gestures[-1])
    else:
        processed_gestures.extend(filtered_gestures[-count:])

    # 공백 제거 및 연속된 중복 제거 (2개 이상 반복되는 경우 하나로 압축)
    final_gestures = []
    for gesture in processed_gestures:
        if not final_gestures or final_gestures[-1] != gesture:
            final_gestures.append(gesture)

    return merge_gestures_to_syllables(final_gestures)

# 초성, 중성, 종성을 조합하여 음절을 만드는 함수
def combine_jamos_to_syllable(cho, jung, jong=None):
    CHO_BASE = 0x1100
    JUNG_BASE = 0x1161
    JONG_BASE = 0x11A7
    SYLLABLE_BASE = 0xAC00

    try:
        cho_index = ord(cho) - CHO_BASE
        jung_index = ord(jung) - JUNG_BASE
        jong_index = ord(jong) - JONG_BASE if jong else 0

        # 인덱스의 유효성 확인
        if cho_index < 0 or jung_index < 0 or (jong and jong_index < 0):
            raise InvalidJamoError("Invalid jamo character")

        # 한글 음절 유니코드 계산
        syllable_code = SYLLABLE_BASE + (cho_index * 21 * 28) + (jung_index * 28) + jong_index
        return chr(syllable_code)
    except (ValueError, InvalidJamoError) as e:
        print(f"Error combining jamos: {e}")
        raise

# 제스처 리스트 후처리 함수
def merge_gestures_to_syllables(gestures):
    syllables = []
    buffer = []

    for gesture in gestures:
        # 유효한 한글 자모인지 확인
        if is_jamo(gesture):
            buffer.append(gesture)
        else:
            print(f"Invalid jamo found and skipped: {gesture}")

        # 초성, 중성, 종성을 결합하여 음절 만들기
        if len(buffer) == 3:
            try:
                syllable = combine_jamos_to_syllable(buffer[0], buffer[1], buffer[2])
                syllables.append(syllable)
                buffer = []  # 버퍼 초기화
            except (ValueError, InvalidJamoError):
                syllables.extend(buffer)
                buffer = []

        # 초성과 중성만 있을 때 결합하여 음절 만들기
        elif len(buffer) == 2:
            try:
                syllable = combine_jamos_to_syllable(buffer[0], buffer[1])
                syllables.append(syllable)
                buffer = []  # 버퍼 초기화
            except (ValueError, InvalidJamoError):
                syllables.extend(buffer)
                buffer = []

    # 남은 버퍼 처리
    if buffer:
        syllables.extend(buffer)

    return ''.join(syllables)  # 한글 단어로 반환

#------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4001, debug=False)

