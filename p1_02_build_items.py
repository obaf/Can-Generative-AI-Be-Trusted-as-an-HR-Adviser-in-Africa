"""Paper 1 — step 2: build the ground truth and the prompt bank.

Ground truth
  * WBL 2026 legal-framework data (law as of 1 Oct 2025): 15 HR-relevant items.
  * ILO EPLex coded indicators 2020 edition (latest year, law as of 1 Jul): 3 dismissal-law items.
Economies: all 54 African countries in WBL (UN member states) + 20 comparison economies (10 high-income, 10 other).
Outputs: data/economies.csv, data/ground_truth.csv, data/prompts.csv
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from common import CACHE, DATA, COMPARISON_HIGH_INCOME, COMPARISON_OTHER, NORTH_AFRICA_MENA

# ----------------------------------------------------------------------------------------------
# Item definitions: (item_id, sheet, value column, legal-basis column, type, unit)
WBL_ITEMS = [
    ("WRK_GENDER_EMP", "3. Work", "Wrk_Law_NonDiscEmp_is", "Wrk_Law_NonDiscEmp_LB", "binary", None),
    ("WRK_AGE_REC", "3. Work", "Wrk_Law_NonDiscRec_Age", "Wrk_Law_NonDiscRec_Age_LB", "binary", None),
    ("WRK_MAR_REC", "3. Work", "Wrk_Law_NonDiscRec_Mar", "Wrk_Law_NonDiscRec_Mar_LB", "binary", None),
    ("WRK_FLEX_TIME", "3. Work", "Wrk_Law_ReqFlex_Time", "Wrk_Law_ReqFlex_Time_LB", "binary", None),
    ("PAY_EQUAL_VALUE", "4. Pay", "Pay_Law_EqRemEqVal_is", "Pay_Law_EqRemEqVal_LB", "binary", None),
    ("PAY_NIGHT", "4. Pay", "Pay_Law_NightWrk_is", "Pay_Law_NightWrk_LB", "binary", None),
    ("PAY_HAZARD", "4. Pay", "Pay_Law_Dangerous_Hazard", "Pay_Law_Dangerous_Hazard_LB", "binary", None),
    ("PAY_MINING", "4. Pay", "Pay_Law_EconSect_Mining", "Pay_Law_EconSect_Mining_LB", "binary", None),
    ("PAR_PREG_DISMISS", "6. Parenthood", "Par_Law_NonDismPreg_is", "Par_Law_NonDismPreg_LB", "binary", None),
    ("PAR_MAT_DAYS", "6. Parenthood", "Par_Law_MatLeave_Total", "Par_Law_MatLeave_LB", "numeric", "days"),
    ("PAR_PAT_DAYS", "6. Parenthood", "Par_Law_PatLeave_Total", "Par_Law_PatLeave_LB", "numeric", "days"),
    ("SAF_HARASS_EMP", "1. Safety", "Saf_Law_SexHar_EplCpCr", "Saf_Law_SexHar_EplCpCr_LB", "binary", None),
    ("PEN_EQUAL_AGE", "10. Pension", "Pen_Law_PensAgeEq_Full", "Pen_Law_PensAgeEq_Full_LB", "binary", None),
    ("PEN_AGE_MEN", "10. Pension", "Pen_Law_PensAgeEq_FullM", "Pen_Law_PensAgeEq_Full_LB", "numeric", "years"),
    ("PEN_AGE_WOMEN", "10. Pension", "Pen_Law_PensAgeEq_FullW", "Pen_Law_PensAgeEq_Full_LB", "numeric", "years"),
]
EPLEX_ITEMS = [
    ("EPX_PROBATION", "trial_max", "numeric", "months"),
    ("EPX_NOTICE4", "notice4", "numeric", "months"),
    ("EPX_REDUND4", "payment4_RED", "numeric", "months_wages"),
]
ITEM_DOMAIN = {
    "WRK": "Recruitment & non-discrimination", "PAY": "Pay & working conditions", "PAR": "Parenthood",
    "SAF": "Workplace harassment", "PEN": "Retirement & pensions", "EPX": "Dismissal (EPLex)",
}
SUBNATIONAL = (r"Ontario|Toronto|New York|N\.Y\.|California|Los Angeles|NSW|New South Wales|Sydney|Maharashtra|Mumbai|Delhi|"
               r"Lagos|Kano|São Paulo|Sao Paulo|Rio de Janeiro|Ciudad de México|Mexico City|Nuevo León|Monterrey|Punjab|Sindh|"
               r"Karachi|Lahore|Addis Ababa|Bavaria|Bayern")
NO_LB = ("no applicable provisions", "no restrictions could be located", "n/a", "ns", "insufficient evidence")

# ----------------------------------------------------------------------------------------------
# Question wording. {c} = country name in the prompt language; {date} = legal reference date.
Q = {
    "en": {
        "pre": "You are advising the HR manager of a private-sector company in {c}. Answer from your own knowledge of the law of {c} as in force on {date} (national legislation, including labour and social-security law).",
        "WRK_GENDER_EMP": "Does the law of {c} prohibit discrimination in employment on the basis of gender?",
        "WRK_AGE_REC": "Does the law of {c} prohibit discrimination in recruitment on the basis of age?",
        "WRK_MAR_REC": "Does the law of {c} prohibit discrimination in recruitment on the basis of marital status?",
        "WRK_FLEX_TIME": "Does the law of {c} give employees the right to request flexible working arrangements regarding their working time?",
        "PAY_EQUAL_VALUE": "Does the law of {c} mandate equal remuneration for women and men for work of equal value?",
        "PAY_NIGHT": "Can a woman work at night in the same way as a man under the law of {c}?",
        "PAY_HAZARD": "Can a woman work in a job deemed hazardous in the same way as a man under the law of {c}?",
        "PAY_MINING": "Can a woman work in the mining sector in the same way as a man under the law of {c}?",
        "PAR_PREG_DISMISS": "Does the law of {c} prohibit the dismissal of pregnant workers?",
        "PAR_MAT_DAYS": "In total, how many calendar days of paid leave is a mother entitled to by law in {c} for the birth of a child (maternity leave plus any paid parental leave available to the mother)?",
        "PAR_PAT_DAYS": "In total, how many calendar days of paid leave is a father entitled to by law in {c} for the birth of a child (paternity leave plus any paid parental leave available to the father)?",
        "SAF_HARASS_EMP": "Does {c} have legislation on sexual harassment in employment that provides criminal penalties or civil remedies?",
        "PEN_EQUAL_AGE": "Under the law of {c}, are the ages at which women and men can retire with full pension benefits the same?",
        "PEN_AGE_MEN": "Under the law of {c}, at what age can a man retire with full (not reduced) old-age pension benefits?",
        "PEN_AGE_WOMEN": "Under the law of {c}, at what age can a woman retire with full (not reduced) old-age pension benefits?",
        "EPX_PROBATION": "What is the maximum length of the probationary (trial) period allowed by law in {c} for a regular employee on an open-ended contract?",
        "EPX_NOTICE4": "What statutory notice period must an employer in {c} give when dismissing a regular employee with 4 years of service for reasons other than serious misconduct?",
        "EPX_REDUND4": "What statutory severance (redundancy) pay is owed in {c} to a regular employee with 4 years of service who is dismissed for economic reasons?",
        "bin": 'Reply with one JSON object and nothing else: {"answer": "Yes" | "No" | "Unsure", "confidence": integer 0-100 = probability that your answer is correct, "legal_basis": "name of the statute and article/section you rely on, or unknown"}.',
        "num": 'Reply with one JSON object and nothing else: {"answer": a number in {u} (use 0 if the law provides none) or "Unsure", "confidence": integer 0-100 = probability that your answer is correct, "legal_basis": "name of the statute and article/section you rely on, or unknown"}.',
        "units": {"days": "calendar days", "years": "years", "months": "months", "months_wages": "months of wages"},
    },
    "fr": {
        "pre": "Vous conseillez le responsable des ressources humaines d'une entreprise privée établie {en}. Répondez à partir de vos propres connaissances du droit {de} en vigueur au {date} (législation nationale, y compris le droit du travail et de la sécurité sociale).",
        "WRK_GENDER_EMP": "La loi {de} interdit-elle la discrimination dans l'emploi fondée sur le sexe ?",
        "WRK_AGE_REC": "La loi {de} interdit-elle la discrimination à l'embauche fondée sur l'âge ?",
        "WRK_MAR_REC": "La loi {de} interdit-elle la discrimination à l'embauche fondée sur la situation matrimoniale ?",
        "WRK_FLEX_TIME": "La loi {de} donne-t-elle aux salariés le droit de demander des aménagements flexibles de leur temps de travail ?",
        "PAY_EQUAL_VALUE": "La loi {de} impose-t-elle l'égalité de rémunération entre les femmes et les hommes pour un travail de valeur égale ?",
        "PAY_NIGHT": "Selon la loi {de}, une femme peut-elle travailler la nuit de la même manière qu'un homme ?",
        "PAY_HAZARD": "Selon la loi {de}, une femme peut-elle occuper un emploi jugé dangereux de la même manière qu'un homme ?",
        "PAY_MINING": "Selon la loi {de}, une femme peut-elle travailler dans le secteur minier de la même manière qu'un homme ?",
        "PAR_PREG_DISMISS": "La loi {de} interdit-elle le licenciement des travailleuses enceintes ?",
        "PAR_MAT_DAYS": "Au total, à combien de jours calendaires de congé rémunéré une mère a-t-elle droit selon la loi {de} pour la naissance d'un enfant (congé de maternité plus tout congé parental rémunéré accessible à la mère) ?",
        "PAR_PAT_DAYS": "Au total, à combien de jours calendaires de congé rémunéré un père a-t-il droit selon la loi {de} pour la naissance d'un enfant (congé de paternité plus tout congé parental rémunéré accessible au père) ?",
        "SAF_HARASS_EMP": "Existe-t-il, {en}, une législation sur le harcèlement sexuel au travail prévoyant des sanctions pénales ou des recours civils ?",
        "PEN_EQUAL_AGE": "Selon la loi {de}, les âges auxquels les femmes et les hommes peuvent partir à la retraite avec une pension complète sont-ils identiques ?",
        "PEN_AGE_MEN": "Selon la loi {de}, à quel âge un homme peut-il partir à la retraite avec une pension de vieillesse complète (non réduite) ?",
        "PEN_AGE_WOMEN": "Selon la loi {de}, à quel âge une femme peut-elle partir à la retraite avec une pension de vieillesse complète (non réduite) ?",
        "EPX_PROBATION": "Quelle est la durée maximale de la période d'essai autorisée par la loi {de} pour un salarié en contrat à durée indéterminée ?",
        "EPX_NOTICE4": "Quel préavis légal un employeur établi {en} doit-il donner pour licencier un salarié ayant 4 ans d'ancienneté pour un motif autre qu'une faute grave ?",
        "EPX_REDUND4": "Quelle indemnité légale de licenciement pour motif économique est due, {en}, à un salarié ayant 4 ans d'ancienneté ?",
        "bin": 'Répondez avec un seul objet JSON et rien d\'autre (valeurs de "answer" en anglais) : {"answer": "Yes" | "No" | "Unsure", "confidence": entier 0-100 = probabilité que votre réponse soit correcte, "legal_basis": "nom du texte de loi et de l\'article sur lequel vous vous fondez, ou unknown"}.',
        "num": 'Répondez avec un seul objet JSON et rien d\'autre : {"answer": un nombre en {u} (0 si la loi n\'en prévoit pas) ou "Unsure", "confidence": entier 0-100 = probabilité que votre réponse soit correcte, "legal_basis": "nom du texte de loi et de l\'article sur lequel vous vous fondez, ou unknown"}.',
        "units": {"days": "jours calendaires", "years": "années", "months": "mois", "months_wages": "mois de salaire"},
    },
    "pt": {
        "pre": "Está a aconselhar o gestor de recursos humanos de uma empresa privada {em}. Responda com base no seu próprio conhecimento da lei {de} em vigor em {date} (legislação nacional, incluindo o direito do trabalho e da segurança social).",
        "WRK_GENDER_EMP": "A lei {de} proíbe a discriminação no emprego com base no sexo?",
        "WRK_AGE_REC": "A lei {de} proíbe a discriminação no recrutamento com base na idade?",
        "WRK_MAR_REC": "A lei {de} proíbe a discriminação no recrutamento com base no estado civil?",
        "WRK_FLEX_TIME": "A lei {de} confere aos trabalhadores o direito de solicitar regimes flexíveis quanto ao seu tempo de trabalho?",
        "PAY_EQUAL_VALUE": "A lei {de} impõe a igualdade de remuneração entre mulheres e homens por trabalho de igual valor?",
        "PAY_NIGHT": "Segundo a lei {de}, uma mulher pode trabalhar à noite da mesma forma que um homem?",
        "PAY_HAZARD": "Segundo a lei {de}, uma mulher pode exercer um trabalho considerado perigoso da mesma forma que um homem?",
        "PAY_MINING": "Segundo a lei {de}, uma mulher pode trabalhar no setor mineiro da mesma forma que um homem?",
        "PAR_PREG_DISMISS": "A lei {de} proíbe o despedimento de trabalhadoras grávidas?",
        "PAR_MAT_DAYS": "No total, a quantos dias de calendário de licença remunerada tem direito uma mãe, por lei, {em}, pelo nascimento de um filho (licença de maternidade mais qualquer licença parental remunerada disponível para a mãe)?",
        "PAR_PAT_DAYS": "No total, a quantos dias de calendário de licença remunerada tem direito um pai, por lei, {em}, pelo nascimento de um filho (licença de paternidade mais qualquer licença parental remunerada disponível para o pai)?",
        "SAF_HARASS_EMP": "Existe {em} legislação sobre assédio sexual no emprego que preveja sanções penais ou reparações civis?",
        "PEN_EQUAL_AGE": "Segundo a lei {de}, as idades em que mulheres e homens se podem reformar com pensão completa são iguais?",
        "PEN_AGE_MEN": "Segundo a lei {de}, com que idade um homem se pode reformar com pensão de velhice completa (não reduzida)?",
        "PEN_AGE_WOMEN": "Segundo a lei {de}, com que idade uma mulher se pode reformar com pensão de velhice completa (não reduzida)?",
        "EPX_PROBATION": "Qual é a duração máxima do período experimental permitida pela lei {de} para um trabalhador com contrato por tempo indeterminado?",
        "EPX_NOTICE4": "Que aviso prévio legal deve um empregador {em} dar ao despedir um trabalhador com 4 anos de antiguidade por motivos que não sejam falta grave?",
        "EPX_REDUND4": "Que indemnização legal por despedimento por motivos económicos é devida {em} a um trabalhador com 4 anos de antiguidade?",
        "bin": 'Responda com um único objeto JSON e nada mais (valores de "answer" em inglês): {"answer": "Yes" | "No" | "Unsure", "confidence": inteiro 0-100 = probabilidade de a sua resposta estar correta, "legal_basis": "nome do diploma legal e do artigo em que se baseia, ou unknown"}.',
        "num": 'Responda com um único objeto JSON e nada mais: {"answer": um número em {u} (0 se a lei não prevê) ou "Unsure", "confidence": inteiro 0-100 = probabilidade de a sua resposta estar correta, "legal_basis": "nome do diploma legal e do artigo em que se baseia, ou unknown"}.',
        "units": {"days": "dias de calendário", "years": "anos", "months": "meses", "months_wages": "meses de salário"},
    },
    "es": {
        "pre": "Usted asesora al gerente de recursos humanos de una empresa privada en {c}. Responda con base en su propio conocimiento de la ley de {c} vigente al {date} (legislación nacional, incluido el derecho laboral y de la seguridad social).",
        "WRK_GENDER_EMP": "¿Prohíbe la ley de {c} la discriminación en el empleo por motivos de sexo?",
        "WRK_AGE_REC": "¿Prohíbe la ley de {c} la discriminación en la contratación por motivos de edad?",
        "WRK_MAR_REC": "¿Prohíbe la ley de {c} la discriminación en la contratación por motivos de estado civil?",
        "WRK_FLEX_TIME": "¿Otorga la ley de {c} a los trabajadores el derecho a solicitar modalidades flexibles respecto de su tiempo de trabajo?",
        "PAY_EQUAL_VALUE": "¿Exige la ley de {c} igual remuneración para mujeres y hombres por un trabajo de igual valor?",
        "PAY_NIGHT": "Según la ley de {c}, ¿puede una mujer trabajar de noche de la misma manera que un hombre?",
        "PAY_HAZARD": "Según la ley de {c}, ¿puede una mujer desempeñar un trabajo considerado peligroso de la misma manera que un hombre?",
        "PAY_MINING": "Según la ley de {c}, ¿puede una mujer trabajar en el sector minero de la misma manera que un hombre?",
        "PAR_PREG_DISMISS": "¿Prohíbe la ley de {c} el despido de trabajadoras embarazadas?",
        "PAR_MAT_DAYS": "En total, ¿a cuántos días naturales de licencia remunerada tiene derecho una madre por ley en {c} por el nacimiento de un hijo (licencia de maternidad más cualquier licencia parental remunerada disponible para la madre)?",
        "PAR_PAT_DAYS": "En total, ¿a cuántos días naturales de licencia remunerada tiene derecho un padre por ley en {c} por el nacimiento de un hijo (licencia de paternidad más cualquier licencia parental remunerada disponible para el padre)?",
        "SAF_HARASS_EMP": "¿Tiene {c} legislación sobre acoso sexual en el empleo que prevea sanciones penales o reparaciones civiles?",
        "PEN_EQUAL_AGE": "Según la ley de {c}, ¿son iguales las edades a las que mujeres y hombres pueden jubilarse con pensión completa?",
        "PEN_AGE_MEN": "Según la ley de {c}, ¿a qué edad puede un hombre jubilarse con una pensión de vejez completa (no reducida)?",
        "PEN_AGE_WOMEN": "Según la ley de {c}, ¿a qué edad puede una mujer jubilarse con una pensión de vejez completa (no reducida)?",
        "EPX_PROBATION": "¿Cuál es la duración máxima del período de prueba permitida por la ley de {c} para un trabajador con contrato indefinido?",
        "EPX_NOTICE4": "¿Qué preaviso legal debe dar un empleador en {c} al despedir a un trabajador con 4 años de antigüedad por motivos distintos de una falta grave?",
        "EPX_REDUND4": "¿Qué indemnización legal por despido por causas económicas se debe en {c} a un trabajador con 4 años de antigüedad?",
        "bin": 'Responda con un único objeto JSON y nada más (valores de "answer" en inglés): {"answer": "Yes" | "No" | "Unsure", "confidence": entero 0-100 = probabilidad de que su respuesta sea correcta, "legal_basis": "nombre de la ley y del artículo en que se basa, o unknown"}.',
        "num": 'Responda con un único objeto JSON y nada más: {"answer": un número en {u} (0 si la ley no lo prevé) o "Unsure", "confidence": entero 0-100 = probabilidad de que su respuesta sea correcta, "legal_basis": "nombre de la ley y del artículo en que se basa, o unknown"}.',
        "units": {"days": "días naturales", "years": "años", "months": "meses", "months_wages": "meses de salario"},
    },
    "ar": {
        "pre": "أنت تقدم المشورة لمدير الموارد البشرية في شركة خاصة في {c}. أجب من معرفتك الخاصة بقانون {c} الساري في {date} (التشريعات الوطنية، بما في ذلك قانون العمل وقانون الضمان الاجتماعي).",
        "WRK_GENDER_EMP": "هل يحظر قانون {c} التمييز في العمل على أساس الجنس؟",
        "WRK_AGE_REC": "هل يحظر قانون {c} التمييز في التوظيف على أساس السن؟",
        "WRK_MAR_REC": "هل يحظر قانون {c} التمييز في التوظيف على أساس الحالة الاجتماعية؟",
        "WRK_FLEX_TIME": "هل يمنح قانون {c} العاملين الحق في طلب ترتيبات مرنة بشأن وقت العمل؟",
        "PAY_EQUAL_VALUE": "هل يفرض قانون {c} المساواة في الأجر بين النساء والرجال عن العمل ذي القيمة المتساوية؟",
        "PAY_NIGHT": "بموجب قانون {c}، هل يمكن للمرأة أن تعمل ليلاً بنفس طريقة الرجل؟",
        "PAY_HAZARD": "بموجب قانون {c}، هل يمكن للمرأة أن تعمل في وظيفة تعتبر خطرة بنفس طريقة الرجل؟",
        "PAY_MINING": "بموجب قانون {c}، هل يمكن للمرأة أن تعمل في قطاع التعدين بنفس طريقة الرجل؟",
        "PAR_PREG_DISMISS": "هل يحظر قانون {c} فصل العاملات الحوامل؟",
        "PAR_MAT_DAYS": "كم إجمالي عدد الأيام التقويمية من الإجازة المدفوعة الأجر التي تستحقها الأم بموجب قانون {c} عند ولادة طفل (إجازة الأمومة مضافاً إليها أي إجازة والدية مدفوعة متاحة للأم)؟",
        "PAR_PAT_DAYS": "كم إجمالي عدد الأيام التقويمية من الإجازة المدفوعة الأجر التي يستحقها الأب بموجب قانون {c} عند ولادة طفل (إجازة الأبوة مضافاً إليها أي إجازة والدية مدفوعة متاحة للأب)؟",
        "SAF_HARASS_EMP": "هل لدى {c} تشريع بشأن التحرش الجنسي في العمل ينص على عقوبات جنائية أو تعويضات مدنية؟",
        "PEN_EQUAL_AGE": "بموجب قانون {c}، هل الأعمار التي يمكن للنساء والرجال التقاعد عندها بمعاش كامل متساوية؟",
        "PEN_AGE_MEN": "بموجب قانون {c}، في أي سن يمكن للرجل التقاعد والحصول على معاش شيخوخة كامل (غير مخفض)؟",
        "PEN_AGE_WOMEN": "بموجب قانون {c}، في أي سن يمكن للمرأة التقاعد والحصول على معاش شيخوخة كامل (غير مخفض)؟",
        "EPX_PROBATION": "ما هي المدة القصوى لفترة الاختبار (التجربة) التي يسمح بها قانون {c} للعامل بعقد غير محدد المدة؟",
        "EPX_NOTICE4": "ما هي مدة الإخطار القانونية التي يجب على صاحب العمل في {c} منحها عند فصل عامل لديه 4 سنوات من الخدمة لأسباب غير الخطأ الجسيم؟",
        "EPX_REDUND4": "ما هو تعويض نهاية الخدمة القانوني المستحق في {c} لعامل لديه 4 سنوات من الخدمة يُفصل لأسباب اقتصادية؟",
        "bin": 'أجب بكائن JSON واحد فقط ولا شيء غيره (قيم "answer" باللغة الإنجليزية): {"answer": "Yes" | "No" | "Unsure", "confidence": عدد صحيح 0-100 = احتمال أن تكون إجابتك صحيحة, "legal_basis": "اسم القانون والمادة التي تستند إليها، أو unknown"}.',
        "num": 'أجب بكائن JSON واحد فقط ولا شيء غيره: {"answer": رقم بوحدة {u} (0 إذا لم ينص القانون على ذلك) أو "Unsure", "confidence": عدد صحيح 0-100 = احتمال أن تكون إجابتك صحيحة, "legal_basis": "اسم القانون والمادة التي تستند إليها، أو unknown"}.',
        "units": {"days": "الأيام التقويمية", "years": "السنوات", "months": "الأشهر", "months_wages": "أشهر من الأجر"},
    },
}
DATE = {"en": "1 October 2025", "fr": "1er octobre 2025", "pt": "1 de outubro de 2025", "es": "1 de octubre de 2025", "ar": "1 أكتوبر 2025"}
DATE_EPX = {"en": "1 July {y}", "fr": "1er juillet {y}", "pt": "1 de julho de {y}", "es": "1 de julio de {y}", "ar": "1 يوليو {y}"}

# Legal-working language for the language-effect condition (African countries whose labour law is
# primarily published in a non-English language; bilingual English/other countries are excluded).
LEGAL_LANG = {
    "fr": ["BEN", "BFA", "BDI", "CAF", "TCD", "COM", "COG", "COD", "CIV", "DJI", "GAB", "GIN", "MDG", "MLI", "NER", "SEN", "TGO"],
    "pt": ["AGO", "CPV", "GNB", "MOZ", "STP"],
    "ar": ["DZA", "EGY", "LBY", "MRT", "MAR", "SDN", "TUN"],
    "es": ["GNQ"],
}
# Country names in each prompt language (only needed for the language-condition countries).
NAMES = {
    "BEN": "Bénin", "BFA": "Burkina Faso", "BDI": "Burundi", "CAF": "République centrafricaine", "TCD": "Tchad",
    "COM": "Comores", "COG": "République du Congo", "COD": "République démocratique du Congo", "CIV": "Côte d'Ivoire",
    "DJI": "Djibouti", "GAB": "Gabon", "GIN": "Guinée", "MDG": "Madagascar", "MLI": "Mali", "NER": "Niger",
    "SEN": "Sénégal", "TGO": "Togo", "AGO": "Angola", "CPV": "Cabo Verde", "GNB": "Guiné-Bissau", "MOZ": "Moçambique",
    "STP": "São Tomé e Príncipe", "DZA": "الجزائر", "EGY": "مصر", "LBY": "ليبيا", "MRT": "موريتانيا", "MAR": "المغرب",
    "SDN": "السودان", "TUN": "تونس", "GNQ": "Guinea Ecuatorial",
}

# Grammatical country forms for French ({de}/{en}) and Portuguese ({de}/{em}) prompts.
FORMS = {
    "BEN": ("du Bénin", "au Bénin"), "BFA": ("du Burkina Faso", "au Burkina Faso"), "BDI": ("du Burundi", "au Burundi"),
    "CAF": ("de la République centrafricaine", "en République centrafricaine"), "TCD": ("du Tchad", "au Tchad"),
    "COM": ("des Comores", "aux Comores"), "COG": ("de la République du Congo", "en République du Congo"),
    "COD": ("de la République démocratique du Congo", "en République démocratique du Congo"),
    "CIV": ("de la Côte d'Ivoire", "en Côte d'Ivoire"), "DJI": ("de Djibouti", "à Djibouti"), "GAB": ("du Gabon", "au Gabon"),
    "GIN": ("de la Guinée", "en Guinée"), "MDG": ("de Madagascar", "à Madagascar"), "MLI": ("du Mali", "au Mali"),
    "NER": ("du Niger", "au Niger"), "SEN": ("du Sénégal", "au Sénégal"), "TGO": ("du Togo", "au Togo"),
    "AGO": ("de Angola", "em Angola"), "CPV": ("de Cabo Verde", "em Cabo Verde"), "GNB": ("da Guiné-Bissau", "na Guiné-Bissau"),
    "MOZ": ("de Moçambique", "em Moçambique"), "STP": ("de São Tomé e Príncipe", "em São Tomé e Príncipe"),
}

# Legal-origin fills where QoG (La Porta et al.) is missing — documented in assumptions.docx.
LEGOR_FILL = {"ETH": 2, "SSD": 1, "ERI": 2, "CPV": 2, "STP": 2, "GNQ": 2, "SYC": 1, "COM": 2, "DJI": 2,
              "LBY": 2, "SOM": 1, "GNB": 2, "SDN": 1, "NAM": 1, "LSO": 1, "SWZ": 1, "BWA": 1, "ZWE": 1, "DEU": 4}
WB_TO_ISO = {"ZAR": "COD", "ROM": "ROU", "TMP": "TLS", "KSV": "XKX"}  # legacy World Bank codes → ISO 3166
LEGOR_LABEL = {1: "English common law", 2: "French civil law", 3: "Socialist", 4: "German civil law", 5: "Scandinavian"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


EPLEX_NAME_TO_ISO = {  # names as they appear in EPLex (after stripping _B/_W and size suffixes)
    "Congo, Democratic Republic": "COD", "Cote d'Ivoire": "CIV", "Tanzania, United Republic of": "TZA",
    "Egypt": "EGY", "Viet Nam": "VNM", "Korea, Republic of": "KOR", "Canada (Federal only)": "CAN",
    "Iran, Islamic Republic of": "IRN", "Moldova, Republic of": "MDA", "Venezuela, Bolivarian Republic of": "VEN",
    "Syrian Arab Republic": "SYR", "Russian Federation": "RUS", "United States": "USA", "United Kingdom": "GBR",
}


def read_sheet(wb_path, sheet):
    raw = pd.read_excel(wb_path, sheet_name=sheet, header=None)
    codes = raw.iloc[1].tolist()
    base = raw.iloc[2].tolist()
    cols = [c if isinstance(c, str) and c.strip() else b for c, b in zip(codes, base)]
    df = raw.iloc[3:].copy()
    df.columns = [str(c) for c in cols]
    df = df.rename(columns={"Economy": "economy", "Economy Code": "iso3", "Region": "region", "Income Group": "income"})
    df["iso3"] = df["iso3"].replace(WB_TO_ISO)  # WBL uses legacy World Bank codes for a few economies
    return df[df["economy"].notna()]


def parse_binary(v):
    s = str(v).strip().lower()
    if s in ("yes", "1", "1.0"):
        return 1
    if s in ("no", "0", "0.0"):
        return 0
    return np.nan  # partial scores (e.g. 0.5) or N/A → ambiguous, dropped


def parse_number(v):
    try:
        x = float(str(v).strip())
        return x if np.isfinite(x) else np.nan
    except ValueError:
        m = re.search(r"\d+(\.\d+)?", str(v))
        return float(m.group()) if m else np.nan


def main():
    wbl_path = CACHE / "wbl26.xlsx"
    work = read_sheet(wbl_path, "3. Work")
    eco = work[["economy", "iso3", "region", "income"]].drop_duplicates()
    africa = eco[(eco.region == "Sub-Saharan Africa") | (eco.economy.isin(NORTH_AFRICA_MENA))].copy()
    africa["group"] = "Africa"
    comp = eco[eco.economy.isin(COMPARISON_HIGH_INCOME + COMPARISON_OTHER)].copy()
    comp["group"] = np.where(comp.economy.isin(COMPARISON_HIGH_INCOME), "High-income comparison", "Other LMIC/UMIC comparison")
    econ = pd.concat([africa, comp], ignore_index=True)
    assert len(africa) == 54, f"expected 54 African economies, got {len(africa)}"
    missing = set(COMPARISON_HIGH_INCOME + COMPARISON_OTHER) - set(comp.economy)
    assert not missing, f"comparison economies not found in WBL: {missing}"

    # ---- covariates
    wdi = pd.read_csv(DATA / "wdi_latest.csv")
    econ = econ.merge(wdi[["iso3", "NY.GDP.PCAP.PP.KD", "IT.NET.USER.ZS", "SP.POP.TOTL"]], on="iso3", how="left")
    qog = pd.read_csv(CACHE / "qog_std_cs_jan22.csv", low_memory=False, encoding="latin-1", usecols=["ccodealp", "lp_legor"])
    econ = econ.merge(qog.rename(columns={"ccodealp": "iso3"}), on="iso3", how="left")
    econ["legor_filled"] = econ.lp_legor.isna() & econ.iso3.isin(LEGOR_FILL)
    econ["lp_legor"] = econ.lp_legor.fillna(econ.iso3.map(LEGOR_FILL))
    econ["legal_origin"] = econ.lp_legor.map(LEGOR_LABEL)
    langs = pd.read_csv(DATA / "wikidata_official_languages.csv")
    econ = econ.merge(langs.groupby("iso3").official_language.apply(lambda s: "; ".join(sorted(set(s)))).rename("official_languages"),
                      on="iso3", how="left")
    econ["english_official"] = econ.official_languages.fillna("").str.contains("English")
    econ["prompt_lang"] = "en"
    for lg, isos in LEGAL_LANG.items():
        econ.loc[econ.iso3.isin(isos), "prompt_lang"] = lg
    econ.to_csv(DATA / "economies.csv", index=False)

    # ---- WBL ground truth
    gt = []
    sheets = {s: read_sheet(wbl_path, s) for s in {i[1] for i in WBL_ITEMS}}
    for item, sheet, vcol, lbcol, typ, unit in WBL_ITEMS:
        df = sheets[sheet].set_index("iso3")
        for iso in econ.iso3:
            v, lb = df.at[iso, vcol], df.at[iso, lbcol]
            truth = parse_binary(v) if typ == "binary" else parse_number(v)
            if typ == "numeric" and item.startswith("PAR") and str(v).strip().upper() in ("N/A", "NS", "0"):
                truth = 0.0  # no paid leave → 0 days
            lb_s = "" if pd.isna(lb) else str(lb).strip()
            gt.append(dict(iso3=iso, item_id=item, source="WBL 2026", type=typ, unit=unit, truth=truth, raw_value=str(v),
                           legal_basis=lb_s, has_statute=bool(lb_s) and not lb_s.lower().startswith(NO_LB),
                           ref_date="1 October 2025"))

    # ---- EPLex ground truth (latest year per country; white-collar / larger-firm variants)
    ep = pd.read_excel(CACHE / "eplex2020.xlsx", sheet_name="Data")
    ep["base"] = ep.country.str.replace(r"\s*_[BW]$", "", regex=True).str.replace(r"\s*\d+[+-]$", "", regex=True).str.strip()
    ep["variant"] = ep.country.str.extract(r"(_[BW]|\d+[+-])$", expand=False).fillna("")
    ep = ep[ep.variant.isin(["", "_W", "15+", "10+"])]
    ep = ep.sort_values("year").groupby("base").tail(1)
    name_to_iso = {norm(n): i for n, i in zip(eco.economy, eco.iso3)}
    for n, i in EPLEX_NAME_TO_ISO.items():
        name_to_iso[norm(n)] = i
    ep["iso3"] = ep.base.map(lambda n: name_to_iso.get(norm(n)))
    unmatched = ep[ep.iso3.isna()].base.tolist()
    ep = ep[ep.iso3.isin(econ.iso3) & (ep.iso3 != "CAN")]  # EPLex codes Canada's federal law only -> ambiguous, excluded
    for item, col, typ, unit in EPLEX_ITEMS:
        for _, r in ep.iterrows():
            gt.append(dict(iso3=r.iso3, item_id=item, source="ILO EPLex 2020", type=typ, unit=unit,
                           truth=parse_number(r[col]), raw_value=str(r[col]), legal_basis="", has_statute=False,
                           ref_date=f"1 July {int(r.year)}", eplex_year=int(r.year)))
    gt = pd.DataFrame(gt)
    gt["domain"] = gt.item_id.str.split("_").str[0].map(ITEM_DOMAIN)
    # WBL codes the law of the main business city: flag rows whose legal basis cites sub-national law
    # (prompts ask about national law, so these rows are excluded from the primary analysis).
    gt["subnational"] = gt.legal_basis.fillna("").str.contains(SUBNATIONAL, regex=True)
    before = len(gt)
    # 'Are pension ages equal?' is unanswerable where no pension-age law exists (coded 0 with 'No applicable
    # provisions'): treat as ambiguous, like partial scores.
    no_system = (gt.item_id == "PEN_EQUAL_AGE") & gt.legal_basis.fillna("").str.lower().str.startswith("no applicable")
    gt.loc[no_system, "truth"] = np.nan
    gt = gt[gt.truth.notna()].reset_index(drop=True)
    gt.to_csv(DATA / "ground_truth.csv", index=False)

    # ---- prompts: English for all; legal-language variant for listed African countries
    prompts = []
    en_name = dict(zip(econ.iso3, econ.economy.replace({"Egypt, Arab Rep.": "Egypt", "Congo, Dem. Rep.": "the Democratic Republic of the Congo",
                                                        "Congo, Rep.": "the Republic of the Congo", "Gambia, The": "The Gambia"})))
    lang_of = dict(zip(econ.iso3, econ.prompt_lang))
    for _, g in gt.iterrows():
        langs_needed = ["en"] + ([lang_of[g.iso3]] if lang_of[g.iso3] != "en" else [])
        for lg in langs_needed:
            T = Q[lg]
            c = en_name[g.iso3] if lg == "en" else NAMES[g.iso3]
            date = DATE[lg] if g.source.startswith("WBL") else DATE_EPX[lg].format(y=int(g.eplex_year))
            fmt = T["bin"] if g.type == "binary" else T["num"].replace("{u}", T["units"][g.unit])
            de, en = FORMS.get(g.iso3, (f"de {c}", f"en {c}"))
            kw = dict(c=c, date=date, de=de, en=en, em=en)
            text = "\n\n".join([T["pre"].format(**kw), T[g.item_id].format(**kw), fmt])
            prompts.append(dict(prompt_id=f"{g.iso3}|{g.item_id}|{lg}", iso3=g.iso3, item_id=g.item_id, lang=lg, prompt=text))
    prompts = pd.DataFrame(prompts)
    prompts.to_csv(DATA / "prompts.csv", index=False)

    summary = {
        "economies": int(len(econ)), "african": int((econ.group == "Africa").sum()),
        "gt_rows_before_dropping_ambiguous": before, "gt_rows": int(len(gt)),
        "gt_by_item": gt.item_id.value_counts().to_dict(),
        "eplex_economies": int(ep.iso3.nunique()), "eplex_african": int(ep.iso3.isin(africa.iso3).sum()),
        "eplex_unmatched_names": unmatched,
        "prompts_total": int(len(prompts)), "prompts_by_lang": prompts.lang.value_counts().to_dict(),
        "subnational_rows": int(gt.subnational.sum()),
        "legal_origin_filled": econ.loc[econ.legor_filled, "iso3"].tolist(),
        "missing_covariates": econ.loc[econ[["NY.GDP.PCAP.PP.KD", "IT.NET.USER.ZS", "lp_legor"]].isna().any(axis=1), "iso3"].tolist(),
    }
    (DATA / "build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
