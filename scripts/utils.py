import pandas as pd 
import ast
arabic_letters = {
    'A': 'أ',
    'B': 'ب',
    'C': 'ج',
    'D': 'د',
    'E': 'ه',
    'F': 'و',
    'G': 'ز',
    'H': 'ح',
    'I': 'ط',
    'J': 'ي',
    'K': 'ك',
    'هـ': 'ه',
    'ا': 'أ'
}   

def translate_numbers(text: str) -> str:
    english_to_arabic = {
        '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
        '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
    }
    
    translation_table = str.maketrans(english_to_arabic)
    return text.translate(translation_table)


def mcq_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    if len(pred) > 2 and pred[0] == '(' and pred[2] == ')':
        pred = pred[1]
    if len(gt) > 2 and gt[0] == '(' and gt[2] == ')':
        gt = gt[1]
    pred = pred[0]
    gt = gt[0]
    pred = arabic_letters.get(pred, pred)
    gt = arabic_letters.get(gt, gt)
    return pred == gt

def create_options_prompt(row_data, option_candidate):
    available_keys = set(row_data.keys()) & set(option_candidate)
    options = {cand: row_data[cand] for cand in available_keys if row_data[cand]}
    sorted_options = dict(sorted(options.items()))
    options_prompt = f"هناك عدة خيارات:\n"
    for key, item in sorted_options.items():
        if pd.notna(item) and item != "nan":
            arabic_key = arabic_letters[key]
            options_prompt += f"{arabic_key}. {item}\n"
    return options_prompt.rstrip("\n")

def mmbench_doc_to_text(doc):
    option_candidate = ["A", "B", "C", "D", "E"]
    options = create_options_prompt(doc, option_candidate)
    question = f"{doc['hint']} {doc['question']} {options}" if pd.notna(doc["hint"]) and doc["hint"] != "nan" else f"{doc['question']} {options}"
    return f"{question}\nأجب بحرف الخيار من الاختيارات المعطاة مباشرة."

def mmbench_eval(pred, gt):
    return mcq_eval(pred, gt)

def mme_doc_to_text(doc):
    question = doc["question"].strip()
    return question

def mme_eval(pred: str, gt: str):
    pred = pred.strip()
    if pred == "صح":
        pred = 'نعم'
    return pred == gt
    
def default_eval(pred: str, gt: str):
    pred = pred.strip()
    gt = gt.strip()
    return pred == gt

def iconqa_options_to_str(options_prompt):
    option_prompt_str = ""
    for i, option in enumerate(options_prompt):
        option_choice = chr(ord("A") + i)
        option_choice = arabic_letters[option_choice]
        option_prompt_str += f"{option_choice}. {option}\n"

    option_prompt_str = option_prompt_str.rstrip("\n")
    return option_prompt_str

iconqa_statement = "بالنظر إلى مجموعة من الصور وسؤال، يرجى تقديم الإجابة على السؤال.\n"
iconqa_options_statement = "السؤال: {question}.\nالخيارات:\n{options}\nالرجاء الإجابة بحرف الخيار من الاختيارات المعطاة مباشرة."
iconqa_freeform_statement = "السؤال: {question}.\nالرجاء الإجابة على السؤال باستخدام كلمة واحدة أو عبارة قصيرة."

def iconqa_doc_to_text(doc):
    question = doc["question"]
    ques_type = doc["ques_type"]
    options_prompt = []

    if ques_type == "choose_img":
        options_prompt.append("The first image.")
        options_prompt.append("The second image.")

        options_str = iconqa_options_to_str(options_prompt)
        full_prompt = f"{iconqa_statement}{iconqa_options_statement.format(question=question, options=options_str)}"

    elif ques_type == "choose_txt":
        choices = doc["choices"].split(",")
        for i, choice in enumerate(choices):
            options_prompt.append(f"{choice}")

        options_str = iconqa_options_to_str(options_prompt)
        full_prompt = f"{iconqa_statement}{iconqa_options_statement.format(question=question, options=options_str)}"

    elif ques_type == "fill_in_blank":
        full_prompt = f"{iconqa_statement}{iconqa_freeform_statement.format(question=question)}"

    return full_prompt

def mmmu_parse_options(options):
    option_letters = [arabic_letters[chr(ord("A") + i)] for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


MMMU_MULTI_CHOICE_PROMPT = "أجب بحرف الخيار من الاختيارات المعطاة مباشرة."
MMMU_OPEN_ENDED_PROMPT = "أجب عن السؤال باستخدام كلمة أو عبارة واحدة."

def mmmu_doc_to_text(doc):
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = mmmu_parse_options(ast.literal_eval(doc["options"].replace("،", ",")))
        question = f"{question}\n{parsed_options}\n\n{MMMU_MULTI_CHOICE_PROMPT}"
    else:
        question = f"{question}\n\n{MMMU_OPEN_ENDED_PROMPT}"
    return question

def mmmu_eval(pred, gt):
    return mcq_eval(pred, gt)

def gqa_doc_to_text(doc):
    question = doc["question"]
    post_prompt = "\nأجب عن السؤال باستخدام كلمة أو عبارة واحدة."
    return f"{question}{post_prompt}"

def gqa_eval(pred, gt):
    return default_eval(pred, gt)

def realworldqa_doc_to_text(doc):
    question = doc["question"].strip()
    pre_prompt = "المستخدم\nالسؤال: "
    return f"{pre_prompt}{question}"

def realworldqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def vqav2_doc_to_text(doc):
    post_prompt = "\nأجب على السؤال باستخدام كلمة أو عبارة واحدة."
    return f"{doc['question']}{post_prompt}"

def vizwiz_vqa_doc_to_text(doc):
    post_prompt = "\nعندما تكون المعلومات المقدمة غير كافية، أجب بـ 'لا يمكن الإجابة'.\nأجب عن السؤال باستخدام كلمة واحدة أو عبارة قصيرة."
    text = f"{doc['question'].capitalize()}{post_prompt}"
    return text

def vizwiz_eval(pred: str, gt: str):
    try:
        _ = ast.literal_eval(gt)
        gt = gt.replace(" ", ", ")
        gt = ast.literal_eval(gt)
        print(gt)
    except:
        gt = gt.strip()
    pred = pred.strip()
    if pred == gt:
        return True
    for x in gt:
        if x in pred:
            return True
    return False

def pope_doc_to_text(doc):
    question = doc["question"].strip()
    return f"{question}\nأجب عن السؤال باستخدام كلمة واحدة أو عبارة قصيرة."

def pope_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    return gt in pred

def countbench_doc_to_text(_):
    return "كم عدد الأشياء الموجودة في الصورة؟\nأجب برقم فقط."

def countbench_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    return translate_numbers(pred) == translate_numbers(gt)

def diagramsMMMU_eval(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if len(gt) == 1:
        return pred[0] == gt
    pred = translate_numbers(pred)
    gt = translate_numbers(gt)
    return gt in pred

def diagramsmmmu_doc_to_text(doc):
    return mmmu_doc_to_text(doc)

def medicalmmmu_eval(pred, gt):
    return mcq_eval(pred, gt)

def medicalmmmu_doc_to_text(doc):
    return mmmu_doc_to_text(doc)

def mmt_doc_to_text(doc):
    question_text = "سؤال: <image>\n" + doc["question"].strip()

    options = []
    for option in ["A", "B", "C", "D", "E", "F", "G", "H", "I"]:
        option_text = doc.get(option)
        if option_text and option_text.strip():
            options.append(f"{arabic_letters[option]}: {option_text.strip()}")

    options_text = "\n".join(options) if options else ""

    formatted_question = f"{question_text}\n{options_text}"
    post_prompt = "\nأجب عن السؤال باستخدام حرف واحد من الخيارات المعطاة."
    formatted_question = f"{formatted_question}{post_prompt}"

    return formatted_question

def medicalmmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def medicalmmt_eval(pred, gt):
    return mcq_eval(pred, gt)

def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"أ. {doc['choice_a']}\n"
    question += f"ب. {doc['choice_b']}\n"
    question += f"ج. {doc['choice_c']}\n"
    question += f"د. {doc['choice_d']}"
    return f"{question}\nأجب بحرف الخيار من الاختيارات المعطاة مباشرة."

def seed_eval(pred, gt):
    return mcq_eval(pred, gt)

def hallucinationmmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def hallucinationmmt_eval(pred, gt):
    return mcq_eval(pred, gt)

def vqammt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def vqammt_eval(pred, gt):
    return mcq_eval(pred, gt)



def mutliimagemmt_doc_to_text(doc):
    return mmt_doc_to_text(doc)

def mutliimagemmt_eval(pred, gt):
    return mcq_eval(pred, gt)


def our_options_to_str(options):
    option_prompt_str = ""
    for i, option in enumerate(options):
        option_choice = chr(ord("A") + i)
        option_choice = arabic_letters[option_choice]
        option_prompt_str += f"{option_choice}. {option}\n"

    option_prompt_str = option_prompt_str.rstrip("\n")
    return option_prompt_str

def our_doc_to_text(doc):
    question_text = "سؤال:\n" + doc["question"].strip()
    options = our_options_to_str(doc["options"])
    options_text = "\n".join(options) if options else ""
    formatted_question = f"{question_text}\n{options_text}"
    post_prompt = "\nأجب عن السؤال باستخدام حرف واحد من الخيارات المعطاة."
    formatted_question = f"{formatted_question}{post_prompt}"
    return formatted_question


def isidocvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def isidocvqa_eval(pred, gt):
    return mcq_eval(pred, gt) 

def patddocvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def patddocvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def celebvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def celebvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def countriesvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def countriesvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def foodvqa_doc_to_text(doc):
    return our_doc_to_text(doc)

def foodvqa_eval(pred, gt):
    return mcq_eval(pred, gt)

def objectcoco_doc_to_text(doc):
    return doc['question']

def objectcoco_eval(pred, gt):
    return mcq_eval(pred, gt)

def blink_doc_to_text(doc):
    return doc['question']

def blink_eval(pred, gt):
    return mcq_eval(pred, gt)