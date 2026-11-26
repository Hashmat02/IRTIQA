#!/usr/bin/env python3
"""code_switch.py — Multilingual transformation pipeline for IRTIQA benchmark evaluation.

Generates translated and code-switched variants of DROP, SNIPS, SkillsBench,
and AppWorld. It can run against a local vLLM model or a hosted
OpenAI-compatible vLLM server.

Setup:
    pip install "vllm>=0.8.0" tqdm

Usage:
    python code_switch.py <dataset> <task> <lang> [options]

    dataset : drop | snips | skillsbench | appworld
    task    : translate | code_switch
    lang    : hi | bn | ur | zh | vi | ne | fa

Examples:
    # Translate DROP to Hindi
    python code_switch.py drop translate hi \
        --input "$IRTIQA_ROOT/Benchmarks/drop/drop_10k.jsonl"

    # Code-switch SNIPS to Urdu-English (processes train/dev/test automatically)
    python code_switch.py snips code_switch ur \
        --input "$IRTIQA_ROOT/Benchmarks/SNIPS"

    # Translate SkillsBench instruction files to Chinese
    python code_switch.py skillsbench translate zh \
        --input "$IRTIQA_ROOT/Benchmarks/skillsbench"

    # Quick smoke-test with first 20 items
    python code_switch.py drop translate hi --limit 20
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IRTIQA_ROOT = Path(os.environ.get("IRTIQA_ROOT", Path(__file__).resolve().parents[2]))
MODEL_PATH = os.environ.get("MODEL_PATH", "google/gemma-4-31b-it")

LANGUAGES: dict[str, str] = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "zh": "Chinese (Simplified Mandarin)",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "fa": "Persian",
}

# Default input paths (overridden by --input)
DEFAULT_INPUT = {
    "drop": str(IRTIQA_ROOT / "Benchmarks/drop/drop_10k.jsonl"),
    "snips": str(IRTIQA_ROOT / "Benchmarks/SNIPS"),
    "skillsbench": str(IRTIQA_ROOT / "Benchmarks/skillsbench"),
    "appworld": str(IRTIQA_ROOT / "Benchmarks/appworld/appworld_tasks.jsonl"),
}

SNIPS_SPLITS = ("train", "dev", "test")

# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

_TRANSLATION_BASE = """You are an expert translator and NLP data annotation specialist.
Translate the following English text into {lang_name}.

YOUR GOAL: produce output that is 100% in {lang_name} script.
No English words. No Latin characters. Everything in {lang_name}.

━━━ EVERY WORD FALLS INTO ONE OF THESE CATEGORIES ━━━

CONTENT WORDS (nouns, verbs, adjectives, adverbs) → TRANSLATE to {lang_name}.
Use real {lang_name} words, not phonetic copies of English:
    animated → متحرک | infantry → پیادہ فوج | militancy → عسکریت پسندی
    comedy → مزاحیہ | forecast → پیشگوئی | recession → کساد بازاری

PROPER NOUNS (people, places, orgs, teams, brands, titles, events) → TRANSLITERATE
into {lang_name} script. Do NOT leave them in Latin:
    Tim Tebow → ٹم ٹیبو | New York City → نیو یارک سٹی | Google → گوگل
    Communist Party → کمیونسٹ پارٹی | Patriots → پیٹریاٹس | Netflix → نیٹ فلکس
    Great Depression → عظیم کساد بازاری | Mann Theatres → مان تھیٹرز
For Chinese: use standard Chinese equivalents (纽约市, 谷歌, 爱国者队, 大萧条, 共产党).
For Vietnamese: proper nouns keep their Latin form naturally (Vietnamese uses Latin script).

NUMERALS → Convert to {lang_name} numeral script where one exists:
    Urdu: 1→۱ 2→۲ 3→۳ … 9→۹ | Hindi: 1→१ 2→२ 3→३ … 9→९
    Chinese/Vietnamese: keep Arabic digits. ALL unit words must be translated.

KEEP IN LATIN (strict exceptions — nothing else):
    • NLU intent/slot tags: SearchScreeningEvent, B-genre, O
    • Code, JSON keys, commands, file paths, URLs, ID strings: /root/input/file.json
    • Pure computing protocols: API, HTML, URL, SQL, HTTP

━━━ ADDITIONAL RULES ━━━
    • NO English glosses in parentheses. Write {lang_name} only — no "(accessibility violations)" after Urdu text.
    • Adjective forms of proper nouns → translate: communist(adj.)→کمیونسٹ, socialist→سوشلسٹ
    • Descriptive org sub-units → translate: "Unemployed Councils"→بے روزگار کونسلیں
    • Titles of works → transliterate or translate: "Conduct Unbecoming"→"غیر مناسب طرزِ عمل"

━━━ SELF-CHECK before returning ━━━
Scan your output for any Latin letters. For each: is it a code/tag/URL exception?
If not, convert it to {lang_name} right now. Return only after this check is complete.

Return ONLY the translated text — no explanation, no source text, no preamble."""

_TRANSLATION_ADDENDA: dict[str, str] = {
    "hi": """HINDI-SPECIFIC: Devanagari script, standard Khari Boli, formal register (आप/वे).
NUMERALS: ALL digits → Devanagari. NEVER use Arabic digits.
    0→० 1→१ 2→२ 3→३ 4→४ 5→५ 6→६ 7→७ 8→८ 9→९ (24→२४, 1929→१९२९)
REQUIRED translations (use these — do not transliterate phonetically):
    [Political/social] communist(adj.)→कम्युनिस्ट | socialist(adj.)→समाजवादी
    militancy→उग्रवाद | solidarity→एकजुटता | unemployment→बेरोजगारी
    inflation→मुद्रास्फीति | recession→मंदी | revolution→क्रांति
    council→परिषद | committee→समिति | union→संघ | strike→हड़ताल
    infantry→पैदल सेना | battalion→बटालियन | regiment→रेजिमेंट
    anti-(prefix)→विरोधी- | division(military)→प्रभाग
    [Sport] offense→आक्रमण पक्ष | defense→रक्षा पक्ष | zone→क्षेत्र
    touchdown→टचडाउन | punt→पंट | fumble→फंबल | field goal→फील्ड गोल
    onside kick→ऑनसाइड किक | yards→गज | sack→सैक | rush→दौड़
    quarterback→क्वार्टरबैक | receiver→रिसीवर | kicker→किकर
    backup→बैकअप | starter→स्टार्टर | season→सीज़न | quarter→तिमाही
    [Entertainment] animated→एनिमेटेड (accepted Hindi industry term) | comedy→हास्य
    horror→डरावनी | thriller→रोमांचक | romance→रोमांटिक | screening→प्रदर्शन
    documentary→वृत्तचित्र | genre→शैली | forecast→पूर्वानुमान""",
    "bn": """BENGALI-SPECIFIC: Write in standard Bengali script (শুদ্ধ ভাষা). Use formal literary form appropriate for academic task descriptions.""",
    "ur": """URDU-SPECIFIC: Nastaliq script (RTL), formal Perso-Arabic vocabulary, SOV word order.
Do not mirror English sentence order. Rebuild each clause in natural Urdu syntax first, then
place transliterated names and required terms inside that Urdu frame.
Keep the Urdu verb at the end of the clause, and keep dates/years where an Urdu writer would
naturally place them rather than drifting them to the end of the sentence.
NUMERALS: ALL digits → Eastern Arabic. NEVER use Western Arabic digits.
    0→۰ 1→۱ 2→۲ 3→۳ 4→۴ 5→۵ 6→۶ 7→۷ 8→۸ 9→۹ (24→۲۴, 1929→۱۹۲۹)
REQUIRED translations (use these — do not transliterate phonetically):
    [Political/social] communist(adj.)→کمیونسٹ | socialist(adj.)→سوشلسٹ
    militancy→عسکریت پسندی | solidarity→یکجہتی | unemployment→بے روزگاری
    inflation→افراطِ زر | recession→کساد بازاری | revolution→انقلاب
    council→کونسل | committee→کمیٹی | union→یونین | strike→ہڑتال
    infantry→پیادہ فوج | battalion→بٹالین | regiment→رجمنٹ
    anti-(prefix)→خلاف- | division(military)→ڈویژن | garrison→چھاؤنی
    [Sport] offense→حملہ | defense→دفاع | zone→علاقہ | pass→پاس
    touchdown→ٹچ ڈاؤن | punt→پنٹ | fumble→فمبل | field goal→فیلڈ گول
    onside kick→آن سائیڈ کِک | yards→گز | sack→ساک | rush→رش
    quarterback→کوارٹر بیک | receiver→ریسیور | kicker→کِکر
    backup→بیک اپ | starter→اسٹارٹر | season→سیزن | quarter→کوارٹر
    [Entertainment] animated→متحرک | comedy→مزاحیہ | horror→خوفناک
    thriller→سنسنی خیز | romance→رومانوی | screening→نمائش
    documentary→دستاویزی | genre→صنف | forecast→پیشگوئی
    [Finance] greenback→ڈالر | inflation→مہنگائی | bond→بانڈ""",
    "zh": """CHINESE-SPECIFIC: Simplified Chinese (普通话), written register (书面语).
Technical terms API/data/software/model may stay English where dominant convention.
REQUIRED translations:
    [Political/social] communist(adj.)→共产主义的 | socialist(adj.)→社会主义的
    militancy→激进主义 | solidarity→团结 | unemployment→失业 | recession→衰退
    council→委员会 | union→工会 | strike→罢工 | inflation→通货膨胀
    infantry→步兵 | battalion→营 | regiment→团 | anti-(prefix)→反-
    [Sport] offense→进攻 | defense→防守 | zone→区域 | pass→传球 | rush→冲跑
    touchdown→触地得分 | punt→弃踢 | fumble→掉球 | sack→擒杀四分卫
    field goal→射门得分 | yards→码 | onside kick→短距离开球
    quarterback→四分卫 | receiver→接球手 | kicker→踢球手
    [Entertainment] animated→动画 | comedy→喜剧 | horror→恐怖 | thriller→惊悚
    documentary→纪录片 | genre→类型 | screening→放映 | forecast→预报
NEVER write the word "Percent" in Latin — use % symbol.""",
    "vi": """VIETNAMESE-SPECIFIC: ALL tone diacritics must be correctly placed — never strip tones.
Northern-dialect conventions.
REQUIRED translations:
    [Political/social] communist(adj.)→cộng sản | socialist(adj.)→xã hội chủ nghĩa
    militancy→chủ nghĩa hiếu chiến | solidarity→đoàn kết | unemployment→thất nghiệp
    council→hội đồng | union→công đoàn | strike→đình công | recession→suy thoái
    infantry→bộ binh | battalion→tiểu đoàn | inflation→lạm phát
    [Entertainment] animated→hoạt hình | comedy→hài kịch | horror→kinh dị
    thriller→ly kỳ | documentary→tài liệu | genre→thể loại | forecast→dự báo
    screening→buổi chiếu | romance→lãng mạn
    [Sport — American football] Note: American football has no standard Vietnamese
    terminology. The following approximations are acceptable:
    touchdown→chạm bóng vào vùng gôn | yard→yard (keep — no standard equivalent)
    field goal→đá thành công | punt→đá bổng | defense→phòng thủ | offense→tấn công
Technical terms (software, AI, model, API, data, playlist, album) may stay English.""",
    "ne": """NEPALI-SPECIFIC: Write in Devanagari script. Formal register for academic descriptions. Keep English technical terms where a natural Nepali equivalent does not exist.""",
    "fa": """PERSIAN-SPECIFIC: Write in Modern Standard Persian, Nastaliq script (RTL). Formal register; maintain verb-final (SOV) word order. Technical terms (software, data, model, API) may stay English.""",
}

def _translation_system(lang: str) -> str:
    base = _TRANSLATION_BASE.format(lang_name=LANGUAGES[lang])
    return base + _TRANSLATION_ADDENDA.get(lang, "")

# For SkillsBench: must preserve fenced code, inline code, file paths
_SKILLSBENCH_TRANSLATION_SYSTEM = """You are a technical translator specialising in software-engineering task descriptions.
Translate the following task instruction from English into {lang_name}.

NEVER TRANSLATE — keep exactly as-is:
    • Fenced code blocks (```…```) — leave 100% unchanged
    • Inline code (`…`) — leave as-is
    • File paths: strings with /root/, ./, or file extensions (.json, .stl, .md, .py, etc.)
    • JSON keys (left side of "key": value pairs)
    • Shell commands, function names, class names, variable names
    • URLs, email addresses, numeric values and units

TRANSLATE (natural-language prose only):
    • Paragraph text that describes what to do
    • Numbered / bulleted list items that are NL instructions
    • Section headers that are NL descriptions

FORMATTING: preserve all Markdown formatting exactly — headers, bullets, bold, code fences.

Return ONLY the translated instruction. No preamble, no trailing explanation."""

def _skillsbench_translation_system(lang: str) -> str:
    return _SKILLSBENCH_TRANSLATION_SYSTEM.format(lang_name=LANGUAGES[lang]) + _TRANSLATION_ADDENDA.get(lang, "")

# For AppWorld: personal-assistant task instructions given to an agent that
# operates apps (Gmail, Amazon, Venmo, Spotify, phone, etc.) via API calls.
# Similar prose shape to SNIPS/DROP, but instructions often embed format
# strings and literal reply text the agent must reproduce verbatim
# downstream (e.g. '<product_name> => $<total_price>', or a quoted
# auto-reply message) -- these must stay intact as *structure*, even though
# any literal English sentence embedded as quoted reply content is still
# translated like any other prose, since a fully-Urdu task should also
# expect the agent to act on it in Urdu.
#
# Deliberately self-contained (does NOT append the shared _TRANSLATION_ADDENDA):
# that addenda tells the model to convert digits to native-script numerals,
# which is right for DROP/SNIPS (read-comprehension spans) but wrong here --
# nearly every number in an AppWorld instruction (a dollar amount, a rating
# threshold, a follower count) is a literal value the agent must parse back
# into a real argument for an API call. A sample run surfaced two concrete
# bugs from getting this wrong: (1) the model conflated dollar amounts like
# "$28" with the <placeholder> token examples below and started emitting
# garbled output like "$<28$"/"$<۲۸>"; (2) it also produced a stray English
# gloss in parens ("اسٹار (star)") because the addenda-less prompt had
# dropped the "no parenthetical glosses" rule that DROP/SNIPS/SkillsBench
# inherit from _TRANSLATION_BASE. Both are fixed explicitly below rather than
# by re-appending the addenda, since the addenda's numeral rule is the
# opposite of what's needed here.
_APPWORLD_TRANSLATION_SYSTEM = """You are an expert translator localizing task instructions for an AI assistant
benchmark. The instructions tell an agent to operate apps (Gmail, Amazon, Venmo, Spotify, phone
contacts, etc.) on a supervisor's behalf by making real API calls. Translate the following
instruction from English into {lang_name}.

CRITICAL — NUMBERS STAY EXACTLY AS WRITTEN, IN WESTERN ARABIC DIGITS:
    Every number in these instructions (a dollar amount, a rating threshold, a follower count, a
    list size) is a literal value the agent must parse back into a real argument for an API call.
    Do NOT convert digits to {lang_name} numerals. Do NOT add brackets, angle brackets, or any
    other decoration around a number. Copy it character-for-character.
    Correct:   "$28" → "$28" (translate the surrounding words only, e.g. "Venmo پر $28 کی درخواست")
    Correct:   "22 followers" → "۲۲" is WRONG — write "22 followers" → "22 فالوورز"
    WRONG:     "$28" → "$<28>" or "$<۲۸>" — never invent brackets around a plain number.

NEVER TRANSLATE — keep exactly as-is:
    • Angle-bracket template placeholders that already appear literally in the source text, e.g.
      <product_name>, <total_price> — these are format-string slots, not real values, and are
      copied verbatim only when the source itself already contains that exact bracketed token.
    • App/API names and technical identifiers: Gmail, Amazon, Venmo, Spotify, SimpleNote.
    • Dates, times, numeric values, and IDs (see rule above).

TRANSLATE (this is the bulk of the text):
    • All natural-language prose describing what the agent should do.
    • Quoted literal reply/message text embedded in the instruction (e.g. an auto-reply the
      agent must send) — translate it fully into {lang_name} too, since a fully-localized task
      expects the agent to act on it in {lang_name}.

STYLE: Formal register, natural {lang_name} grammar and word order — do not mirror English
sentence structure word-for-word. Never add an English word in parentheses after its
{lang_name} translation or transliteration (e.g. do NOT write "اسٹار (star)" — write only "اسٹار").

Return ONLY the translated instruction — no explanation, no preamble."""

def _appworld_translation_system(lang: str) -> str:
    return _APPWORLD_TRANSLATION_SYSTEM.format(lang_name=LANGUAGES[lang])

_APPWORLD_CS_SYSTEM = """You are a code-switching generation expert for NLP research. Rewrite the following
AI-assistant task instruction (telling an agent to operate apps like Gmail, Amazon, Venmo, Spotify,
phone contacts, etc.) in naturalistic {lang_name}-English code-switched style, as a fluent bilingual
speaker would write it.

NEVER TOUCH (keep 100% in English):
    • Angle-bracket template placeholders: <product_name>, <total_price>, etc.
    • App/API names: Gmail, Amazon, Venmo, Spotify
    • Format-string punctuation/structure: "$<total_price>", "=>", separators
    • Dates, times, numeric values, and IDs

Code-switch the natural-language prose per the rules below. App names stay in English regardless
(they are brand names, same as named entities in other rules).

{lang_specific}

Return ONLY the code-switched instruction. No explanation."""

def _appworld_cs_system(lang: str) -> str:
    return _APPWORLD_CS_SYSTEM.format(
        lang_name=LANGUAGES[lang],
        lang_specific=_CS_LANG_RULES.get(lang, ""),
    )

# ---------------------------------------------------------------------------
# Code-switching prompts
# ---------------------------------------------------------------------------

_CS_BASE = """You are a code-switching generation expert for NLP research. Rewrite the following English text in naturalistic {lang_name}-English code-switched style, as a fluent bilingual speaker would write it.

MATRIX / EMBEDDED LANGUAGE ROLES:

{lang_name} is the MATRIX language: it supplies grammar, word order, verb morphology, conjunctions, pronouns, postpositions/particles, AND the majority of content vocabulary.

English supplies ONLY:
    (a) Named entities — people, teams, places, organisations, brand names
    (b) Domain-specific technical jargon with no common {lang_name} equivalent (e.g. NFL positions, financial instruments, medical terms, acronyms, sport-specific terms)
    (c) Loanwords already established in {lang_name} usage

Common content words that have a standard {lang_name} equivalent MUST be in {lang_name}: words like economy, season, member, rate, history, city, game, goal, score, period, winter, summer, attack, defense, plan, report, result, vote, rate, loss, gain, etc.

Apply {lang_name} verb morphology to any English verbs that appear.

Do NOT romanize {lang_name} — use native script only.

Target: at least 50–60% of all content words in {lang_name}.

HARD CONSTRAINTS (never violate):
    • All digit sequences appear exactly as in the source (25, 1929, 3.14).
    • Named entities (people, teams, places, organisations) stay in English.
    • The correct answer / intent must still be derivable from the output.
    • Do not alter meaning, reasoning structure, or slot values.

Return ONLY the code-switched text — no explanation.

{lang_specific}"""

_CS_LANG_RULES: dict[str, str] = {
    "hi": """HINDI-ENGLISH RULES:

Matrix: Hindi (Devanagari). Embedded: English named entities and true jargon only.

Retain Hindi: pronouns (मैं, वो, हम, आप), postpositions (ने, को, में, से, का, की, के), conjunctions (और, लेकिन, तो, क्योंकि, अगर, जब), question words (क्या, कैसे, क्यों, कहाँ, कितना), AND common content words: अर्थव्यवस्था (economy), बेरोजगारी (unemployment), मौसम (season/weather), सदस्य (member), दर (rate), इतिहास (history), शहर (city), खेल (game), लक्ष्य (goal), नुकसान (loss), योजना (plan), परिणाम (result).

English stays for: named entities (people/team/place/org/brand names IN ENGLISH),
sport-specific positions and plays, financial instruments, acronyms, AND:
    • entertainment: movie, film, animated, comedy, horror, thriller, show, screening
    • music: music, song, track, album, playlist, soundtrack, genre, artist
    • weather: weather, forecast, warm, cold, humid, rain, temperature
    • assistant actions: play, find, search, book, rate, add, schedule

English verbs get Hindi auxiliary: "play करें", "book करें", "find करें".
Example: "Are there any animated movies playing at the Mann Theatres?"
    → "क्या Mann Theatres में कोई animated movies play हो रही हैं?"
Example: "What is the weather like at Emma Wood State Beach?"
    → "Emma Wood State Beach पर weather कैसा है?"
Example: "The economy was perilous and unemployment hit 25 percent."
    → "अर्थव्यवस्था बहुत खतरनाक थी और बेरोजगारी की दर 25 प्रतिशत तक पहुँच गई।" """,
    "bn": """BENGALI-ENGLISH RULES:

Matrix: Bengali (Bengali script). Embedded: English nouns/terms.

Retain Bengali: pronouns (আমি, সে, তারা, আপনি), case markers (-এর, -কে, -তে, -থেকে, -দিয়ে), conjunctions (এবং, কিন্তু, তাই, কারণ, যদি, যখন).

English verbs get Bengali morphology: "submit করেছি", "complete করবে", "analyze করা হয়েছে".
Example: "The patient was given medication twice daily."
    → "রোগীকে দিনে দুইবার medication দেওয়া হয়েছিল।" """,
    "ur": """URDU-ENGLISH RULES:

Matrix: Urdu (Nastaliq, RTL). The surface form must read like Urdu with English islands
embedded inside it, never like English with Urdu words sprinkled in.

Keep right-to-left Urdu clause order throughout; do not mirror the source English word order.

Urdu SOV word order applies — the Urdu verb ALWAYS stays at clause end.

If the source sentence begins with an English phrase, rewrite the clause into Urdu order first
and then embed only the allowed English terms.

Retain in Urdu: postpositions (نے، کو، میں، سے، کا، کی، کے), conjunctions (اور، لیکن، کیونکہ، اگر، جب), pronouns (وہ، ہم، میں، آپ), question words (کیا، کیسے، کہاں، کتنا، کب).

English stays for: named entities (team/player/place/org/brand names IN ENGLISH, not transliterated),
sport-specific terms (quarterback, touchdown, field goal, yards, sack, punt, offense, defense),
financial instruments, acronyms, AND the following domain terms to distinguish from translation:
    • entertainment: movie, film, animated, comedy, horror, thriller, show, screening
    • music: music, song, track, album, playlist, soundtrack, genre, artist
    • weather: weather, forecast, warm, cold, humid, rain, temperature
    • assistant actions: play, find, search, book, rate, add, schedule

English verbs get Urdu morphology: "play کریں"، "book کر دیں"، "find کریں".

Keep numeric expressions and years in the same relative spot they would naturally occupy in
Urdu; do not push them to the end of the sentence just because English source order did.
Example: "Are there any animated movies playing at the Mann Theatres?"
    → "کیا Mann Theatres میں کوئی animated movies play ہو رہی ہیں؟"
Example: "What is the weather like at Emma Wood State Beach?"
    → "Emma Wood State Beach پر weather کیسا ہے؟"
Example: "Play music by Susumu Hirasawa on Netflix."
    → "Netflix پر Susumu Hirasawa کی music play کریں۔"
Example: "The economy was perilous and unemployment hit 25 percent."
    → "معیشت بہت خطرناک تھی اور بے روزگاری کی شرح 25 فیصد تک پہنچ گئی۔" """,
    "zh": """CHINESE-ENGLISH RULES:

Matrix: Mandarin Chinese (Simplified). NO morphological blending — isolating language.

Switch only at NOUN-PHRASE boundaries; NEVER insert English verbs — use Chinese verbs always.

Wrong: "stock market 在 October crashed 了"  Right: "stock market 在10月崩盘了"
Wrong: "economy became perilous"  Right: "经济变得非常危险"

Retain Chinese: particles (了, 的, 吗, 呢, 把, 被, 也, 都), measure words (个, 件, 本, 次), pronouns (我, 他, 她, 我们, 你), conjunctions (但是, 因为, 如果, 当), AND all verbs, adjectives, and common nouns (经济, 历史, 季节, 比赛, 结果, 损失, 计划).

English stays for: named entities, sport positions/plays, financial instruments, acronyms, AND:
    • entertainment: movie, film, animated, comedy, horror, thriller, show, screening
    • music: music, song, track, album, playlist, soundtrack, genre, artist
    • weather: weather, forecast, warm, humid, temperature
    • assistant actions: play, find, book, rate, schedule
Example: "Are there any animated movies playing at Mann Theatres?"
    → "Mann Theatres 有什么 animated movies 在放映吗？"
Example: "What is the weather like at Emma Wood State Beach?"
    → "Emma Wood State Beach 的 weather 怎么样？"
Example: "The economy was perilous and unemployment hit 25 percent."
    → "经济形势非常危险，失业率达到了25%。" """,
    "vi": """VIETNAMESE-ENGLISH RULES:

Matrix: Vietnamese (Latin script WITH all tone diacritics — NEVER strip a single accent mark).

NO morphological blending — isolating language; never attach Vietnamese suffixes to English words.

Vietnamese and English both use Latin script — make the mixing VISIBLE by keeping a substantial
number of English content words as-is (without translating them into Vietnamese).

Target: keep approximately 35–45% of content words in English.

Vietnamese provides: grammar particles (của, trong, và, nhưng, vì, nếu, thì, đã, đang, sẽ, rồi), pronouns (tôi, bạn, họ, chúng tôi), conjunctions, postpositions, AND core verbs/adjectives (đi, đến, có, là, được, lớn, nhỏ, tốt, xấu, v.v.).

English stays for: ALL named entities, sport positions AND plays (quarterback, touchdown, field goal, running back, placekicker, scramble, offense, defense, punt, kickoff), economic/financial terms (stock market, unemployment rate, recession, bonds, GDP), technology terms (model, data, API, software, report, playlist, album, track, genre), domain jargon, AND common nouns where English is widely understood (season, score, team, game, match, record, loss, win, goal, plan, member, rate, result).

ALL Vietnamese tone marks are MANDATORY — never write "nguoi" when it should be "người".
Example: "The economy was perilous and unemployment hit 25 percent."
    → "Nền economy rất nguy hiểm và unemployment rate đạt 25 phần trăm."
Example: "Broncos scored a touchdown in the first quarter."
    → "Broncos đã ghi được một touchdown trong hiệp một."
Example: "Play jazz music by Miles Davis on Spotify."
    → "Phát jazz music của Miles Davis trên Spotify." """,
    "ne": """NEPALI-ENGLISH RULES:

Matrix: Nepali (Devanagari). SOV word order.

Retain Nepali: postpositions (ले, लाई, मा, बाट, को, का, की), conjunctions (र, तर, किनभने, यदि, जब), pronouns (म, ऊ, हामी, तपाईं).

English verbs get Nepali morphology: "submit गर्नु", "complete गरियो", "analyze गर्छ".
Example: "If the score is above 90, the candidate passes."
    → "यदि score 90 भन्दा माथि छ भने, candidate pass हुन्छ।" """,
    "fa": """PERSIAN-ENGLISH RULES:

Matrix: Persian (Nastaliq, RTL). SOV word order; verb at clause end.

Retain Persian: postpositions (را، در، از، با، برای، به), conjunctions (و، اما، چون، اگر، پس، وقتی), pronouns (من، او، ما، شما).

English verbs get Persian morphology: "submit کردن"، "complete کرد"، "analyze می‌کند".

Technical terms stay English: software, data, report, model, API, deadline.
Example: "He submitted the report after the meeting."
    → "او بعد از meeting، report را submit کرد۔" """,
}

def _cs_system(lang: str) -> str:
    return _CS_BASE.format(
        lang_name=LANGUAGES[lang],
        lang_specific=_CS_LANG_RULES.get(lang, ""),
    )

_SKILLSBENCH_CS_SYSTEM = """You are a code-switching generation expert for NLP research. Rewrite the following technical task description in naturalistic {lang_name}-English code-switched style, as a fluent bilingual engineer would write it.

Apply standard code-switching rules for {lang_name}-English, PLUS:

NEVER TOUCH (keep 100% in English):
    • Fenced code blocks (```…```) — leave completely unchanged
    • Inline code (`…`) — leave as-is
    • File paths (/root/…, ./…, *.json, *.py, etc.) — leave as-is
    • JSON keys and JSON structure
    • Shell commands, function/class/variable names
    • URLs, numeric values

Code-switch ONLY the natural-language prose portions of the instruction.

{lang_specific}

Return ONLY the code-switched instruction. No explanation."""

def _skillsbench_cs_system(lang: str) -> str:
    return _SKILLSBENCH_CS_SYSTEM.format(
        lang_name=LANGUAGES[lang],
        lang_specific=_CS_LANG_RULES.get(lang, ""),
    )

def get_system_prompt(dataset: str, task: str, lang: str) -> str:
    """Return the appropriate system prompt for a given (dataset, task, lang) triple."""
    if dataset == "skillsbench":
        return _skillsbench_translation_system(lang) if task == "translate" else _skillsbench_cs_system(lang)
    if dataset == "appworld":
        return _appworld_translation_system(lang) if task == "translate" else _appworld_cs_system(lang)
    return _translation_system(lang) if task == "translate" else _cs_system(lang)

# ---------------------------------------------------------------------------
# Code-block protection (SkillsBench)
# ---------------------------------------------------------------------------

_FENCED = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE = re.compile(r"`[^\n]+`")
# Plain-ASCII, bracket-delimited marker -- NUL-byte-wrapped placeholders
# (the previous scheme) get silently stripped of their \x00 bytes by many
# LLM tokenizers/generation pipelines, which breaks the exact-match restore
# below ~30% of the time whenever a task actually contains code/paths to
# protect. Bracket-delimited plain text survives verbatim far more reliably.
_PLACEHOLDER = "[[PROT{}]]"
_PLACEHOLDER_CORE = re.compile(r"PROT(\d+)")

def protect_code(text: str) -> tuple[str, dict[str, str]]:
    """Extract fenced and inline code segments into placeholders."""
    store: dict[str, str] = {}
    counter = [0]

    def _sub(m: re.Match) -> str:
        key = _PLACEHOLDER.format(counter[0])
        store[key] = m.group(0)
        counter[0] += 1
        return key

    text = _FENCED.sub(_sub, text)
    text = _INLINE.sub(_sub, text)
    return text, store

def restore_code(text: str, store: dict[str, str]) -> str:
    for key, val in store.items():
        if key in text:
            text = text.replace(key, val)
            continue
        # Fallback: the model preserved the "PROTn" core but mangled/dropped
        # the surrounding brackets (whitespace, stray punctuation, etc.).
        m = _PLACEHOLDER_CORE.search(key)
        if not m:
            continue
        core_pattern = re.compile(r"[\[\(\{]*\s*PROT" + m.group(1) + r"\s*[\]\)\}]*")
        if core_pattern.search(text):
            text = core_pattern.sub(lambda _m, v=val: v, text, count=1)
    return text

# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------

_PREAMBLE = re.compile(
    r"^(here(?:'s| is) the (translation|code.switched|output|result)|"
    r"(translation|output|result|code.switched text))[:\s]+",
    re.IGNORECASE,
)

def clean(text: str) -> str:
    """Strip common model preambles that sneak into responses."""
    return _PREAMBLE.sub("", text).strip()

_APPWORLD_GLOSS = re.compile(r"\s*\(([a-zA-Z]+)\)")
_APPWORLD_GARBLE = re.compile(r"thể\S*نرا\s*\(genre\)")

def strip_appworld_glosses(source_instruction: str, transformed_instruction: str) -> str:
    """Remove model-added English glosses while preserving source parentheses."""
    source_parens = set(re.findall(r"\(([a-zA-Z]+)\)", source_instruction))

    def _sub(m: re.Match) -> str:
        word = m.group(1)
        if word in source_parens:
            return m.group(0)
        return ""

    return _APPWORLD_GLOSS.sub(_sub, transformed_instruction)

def clean_appworld_instruction(source_instruction: str,
                               transformed_instruction: str,
                               lang: str,
                               task: str) -> str:
    """Apply deterministic AppWorld cleanup observed during dataset generation."""
    if lang == "ur" and task == "translate":
        transformed_instruction = _APPWORLD_GARBLE.sub("صنف", transformed_instruction)
        transformed_instruction = strip_appworld_glosses(source_instruction, transformed_instruction)
    return transformed_instruction

# ---------------------------------------------------------------------------
# vLLM inference — two modes
# (A) Direct: load model in-process via from vllm import LLM  (default)
# (B) API   : call a hosted vLLM OpenAI-compatible server via --api-url
# ---------------------------------------------------------------------------

def load_model(model_path: str, tensor_parallel_size: int = 2, max_model_len: int = 8192,
               api_url: str | None = None):
    """Return a local vLLM LLM object, or None when using the hosted API."""
    if api_url:
        log.info("API mode — connecting to %s (no local model load)", api_url)
        return None
    try:
        from vllm import LLM
    except ImportError:
        log.error("vLLM not found. Install with:  pip install 'vllm>=0.8.0'")
        sys.exit(1)
    log.info("Loading %s (tensor_parallel_size=%d, max_model_len=%d)…",
             model_path, tensor_parallel_size, max_model_len)
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

def _batch_chat_api(api_url: str,
                    api_model: str,
                    conversations: list[list[dict]],
                    temperature: float,
                    max_tokens: int,
                    max_workers: int = 32,
                    ) -> list[str]:
    """Send all conversations concurrently to the hosted vLLM server.
    vLLM queues and batches them internally for GPU efficiency."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package required for API mode:  pip install openai")
        sys.exit(1)

    client = OpenAI(base_url=f"{api_url}/v1", api_key="EMPTY")

    def call_one(idx_conv: tuple[int, list[dict]]) -> tuple[int, str]:
        idx, conv = idx_conv
        resp = client.chat.completions.create(
            model=api_model,
            messages=conv,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return idx, clean(resp.choices[0].message.content)

    results: list[str] = [""] * len(conversations)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(call_one, (i, c)): i for i, c in enumerate(conversations)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  API calls", leave=False):
            i, text = fut.result()
            results[i] = text
    return results

def batch_chat(llm,
               conversations: list[list[dict]],
               temperature: float,
               max_tokens: int,
               api_url: str | None = None,
               api_model: str = MODEL_PATH,
               ) -> list[str]:
    """Dispatch to local vLLM or hosted API depending on mode."""
    if api_url:
        return _batch_chat_api(api_url, api_model, conversations, temperature, max_tokens)
    from vllm import SamplingParams
    params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
    outputs = llm.chat(conversations, sampling_params=params, use_tqdm=False)
    return [clean(o.outputs[0].text) for o in outputs]

def make_conversation(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(records), path)

def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def write_lines(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info("Saved %d lines → %s", len(lines), path)

def chunks(lst: list, size: int) -> Iterator[list]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]

# ---------------------------------------------------------------------------
# DROP handler
# ---------------------------------------------------------------------------

def process_drop(llm,
                 input_path: Path,
                 output_dir: Path,
                 lang: str,
                 task: str,
                 batch_size: int,
                 limit: int | None,
                 temperature: float,
                 max_tokens: int,
                 api_url: str | None = None,
                 api_model: str = MODEL_PATH,
                 ) -> None:
    """Transform DROP passages, questions, and answer spans.

    Output: <output_dir>/drop_10k_{lang}_{task}.jsonl
    Added fields per record:
        passage_{lang}       — transformed passage
        question_{lang}      — transformed question
        answers_spans_{lang} — transformed answer spans (same structure as answers_spans)
    """
    records = load_jsonl(input_path)
    if limit:
        records = records[:limit]

    out_path = output_dir / f"{input_path.stem}_{lang}_{task}.jsonl"

    # Resume: reload any existing output
    if out_path.exists():
        existing = {r["query_id"]: r for r in load_jsonl(out_path)}
        records = [existing.get(r["query_id"], r) for r in records]
        log.info("Resuming: %d already in output", len(existing))

    sys_prompt = get_system_prompt("drop", task, lang)
    p_key = f"passage_{lang}"
    q_key = f"question_{lang}"
    a_key = f"answers_spans_{lang}"

    # --- Pass 1: passage + question ---
    pending = [(i, r) for i, r in enumerate(records) if p_key not in r or q_key not in r]
    log.info("DROP %s→%s: %d records need passage/question", task, lang, len(pending))

    for batch in tqdm(list(chunks(pending, batch_size)), desc=f"drop/{lang}/{task} [psg+q]"):
        convs: list[list[dict]] = []
        for _, r in batch:
            convs.append(make_conversation(sys_prompt, r["passage"]))
            convs.append(make_conversation(sys_prompt, r["question"]))

        responses = batch_chat(llm, convs, temperature, max_tokens, api_url, api_model)

        for j, (i, _) in enumerate(batch):
            records[i][p_key] = responses[j * 2]
            records[i][q_key] = responses[j * 2 + 1]

        save_jsonl(records, out_path)

    # --- Pass 2: answer spans ---
    # Each record may have 1-2 spans; flatten to (record_idx, span_idx) pairs
    pending_spans: list[tuple[int, int]] = []
    for i, r in enumerate(records):
        if a_key not in r:
            for si in range(len(r["answers_spans"]["spans"])):
                pending_spans.append((i, si))

    log.info("DROP %s→%s: %d spans to translate", task, lang, len(pending_spans))

    for batch in tqdm(list(chunks(pending_spans, batch_size)), desc=f"drop/{lang}/{task} [spans]"):
        convs = [
            make_conversation(sys_prompt, records[i]["answers_spans"]["spans"][si])
            for i, si in batch
        ]
        responses = batch_chat(llm, convs, temperature, max_tokens, api_url, api_model)

        for k, (i, si) in enumerate(batch):
            if a_key not in records[i]:
                # First span for this record — initialise structure
                records[i][a_key] = {
                    "spans": [None] * len(records[i]["answers_spans"]["spans"]),
                    "types": records[i]["answers_spans"]["types"],
                }
            records[i][a_key]["spans"][si] = responses[k]

        save_jsonl(records, out_path)

    save_jsonl(records, out_path)
    log.info("DROP done: %s", out_path)

# ---------------------------------------------------------------------------
# SNIPS handler
# ---------------------------------------------------------------------------

def process_snips(llm,
                  input_dir: Path,
                  output_dir: Path,
                  lang: str,
                  task: str,
                  batch_size: int,
                  limit: int | None,
                  temperature: float,
                  max_tokens: int,
                  api_url: str | None = None,
                  api_model: str = MODEL_PATH,
                  ) -> None:
    """Transform SNIPS utterances (seq.in only).

    Output directory structure mirrors source:
        <output_dir>/{lang}_{task}/train/seq.in  (transformed)
                                   train/seq.out  (original — token alignment broken after CS/translation)
                                   train/label    (original — intent labels unchanged)
        … same for dev/ and test/

    NOTE: seq.out slot labels will be token-misaligned after transformation.
    This is an expected and documented limitation: for multilingual agentic evaluation
    we rely on intent (label) accuracy, not slot-level F1. Flag for re-annotation
    if slot-level evaluation is needed.
    """
    sys_prompt = get_system_prompt("snips", task, lang)
    split_out_root = output_dir / f"{lang}_{task}"

    # Detect flat layout (sampled dir: seq.in lives directly in input_dir, no train/dev/test)
    flat = (input_dir / "seq.in").exists()
    splits_to_process = [(".", input_dir)] if flat else [(s, input_dir / s) for s in SNIPS_SPLITS]

    for split_name, split_in in splits_to_process:
        if not split_in.exists():
            log.warning("SNIPS split not found, skipping: %s", split_in)
            continue

        utterances = read_lines(split_in / "seq.in")
        if limit:
            utterances = utterances[:limit]
        log.info("SNIPS %s split=%s: %d utterances", task, split_name, len(utterances))

        # Flat layout writes directly into split_out_root; split layout adds subdir
        out_dir = split_out_root if flat else split_out_root / split_name
        out_seq_in = out_dir / "seq.in"

        # Resume
        transformed: list[str] = []
        if out_seq_in.exists():
            transformed = read_lines(out_seq_in)
        start = len(transformed)
        log.info("  Resuming from utterance %d", start)

        pending = utterances[start:]
        for batch_utts in tqdm(list(chunks(pending, batch_size)), desc=f"snips/{split_name}/{lang}/{task}"):
            convs = [make_conversation(sys_prompt, u) for u in batch_utts]
            responses = batch_chat(llm, convs, temperature, max_tokens, api_url, api_model)
            # SNIPS utterances are single-line by construction (1 utterance ==
            # 1 row in seq.in/seq.out/label). If the model ever emits an
            # embedded newline, write_lines would silently turn 1 utterance
            # into 2+ physical lines, breaking row-alignment with the
            # (untouched) label/seq.out for every row after that point.
            responses = [" ".join(r.split("\n")).strip() for r in responses]
            transformed.extend(responses)
            write_lines(transformed, out_seq_in)

        write_lines(transformed, out_seq_in)

        # Copy unchanged files: seq.out and label
        for fname in ("seq.out", "label"):
            src = split_in / fname
            dst = out_dir / fname
            if src.exists():
                lines = read_lines(src)
                if limit:
                    lines = lines[:limit]
                write_lines(lines, dst)

    # Copy shared label files
    import shutil
    for fname in ("intent_label.txt", "slot_label.txt"):
        src = input_dir / fname
        dst = split_out_root / fname
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log.info("Copied %s → %s", src.name, dst)

    # Write alignment-warning metadata
    meta = {
        "dataset": "SNIPS",
        "lang": lang,
        "task": task,
        "note": (
            "seq.out (slot BIO labels) are copied from the original English split and are "
            "token-misaligned with the transformed seq.in. They are retained for reference only. "
            "For slot-level evaluation, re-annotation is required. "
            "Intent labels (label files) are unchanged and valid for intent classification evaluation."
        ),
    }
    meta_path = split_out_root / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info("SNIPS done → %s", split_out_root)

# ---------------------------------------------------------------------------
# SkillsBench handler
# ---------------------------------------------------------------------------

def process_skillsbench(llm,
                        input_dir: Path,
                        output_dir: Path,
                        lang: str,
                        task: str,
                        batch_size: int,
                        limit: int | None,
                        temperature: float,
                        max_tokens: int,
                        api_url: str | None = None,
                        api_model: str = MODEL_PATH,
                        ) -> None:
    """Translate or code-switch SkillsBench task instruction.md files.

    Output: instruction_{lang}_{task}.md written alongside each task's instruction.md.
    Only the natural-language prose is transformed; fenced code, inline code, and
    file paths are protected via placeholder extraction and restored post-generation.

    SkillsBench note: tasks are complex multi-step agent workflows; the instruction is
    the only component translated — environment, verifier, tests, and task.toml are
    unchanged, preserving task integrity.
    """
    sys_prompt = get_system_prompt("skillsbench", task, lang)

    task_dirs: list[Path] = []
    for sub in ("tasks", "tasks-extra"):
        td = input_dir / sub
        if td.exists():
            task_dirs.extend(sorted(p for p in td.iterdir() if p.is_dir()))

    # Flat layout: sampled dir has task folders directly (no tasks/ subdir)
    if not task_dirs and not any((input_dir / d).is_dir() for d in ("tasks", "tasks-extra")):
        task_dirs = sorted(p for p in input_dir.iterdir()
                           if p.is_dir() and (p / "instruction.md").exists())

    if limit:
        task_dirs = task_dirs[:limit]

    log.info("SkillsBench %s→%s: %d tasks", task, lang, len(task_dirs))

    # Build work list: (task_dir, instruction_text, protected_store, out_path)
    WorkItem = tuple[Path, str, dict[str, str], Path]
    work: list[WorkItem] = []

    for td in task_dirs:
        src = td / "instruction.md"
        if not src.exists():
            log.warning("No instruction.md: %s", td)
            continue
        out_name = f"instruction_{lang}_{task}.md"
        # Always write alongside the source instruction.md in the task dir
        out = td / out_name

        raw = src.read_text(encoding="utf-8")
        protected, store = protect_code(raw)

        if out.exists():
            existing = out.read_text(encoding="utf-8")
            repaired = restore_code(existing, store)
            if repaired != existing:
                out.write_text(repaired, encoding="utf-8")
                log.info("Repaired protected placeholders in existing output: %s", out)
            log.debug("Already done, skipping: %s", out)
            continue

        work.append((td, protected, store, out))

    log.info("  %d tasks need processing (%d already done)", len(work), len(task_dirs) - len(work))

    for batch in tqdm(list(chunks(work, batch_size)), desc=f"skillsbench/{lang}/{task}"):
        convs = [make_conversation(sys_prompt, protected) for (_, protected, _, _) in batch]
        responses = batch_chat(llm, convs, temperature, max_tokens, api_url, api_model)

        for (td, _, store, out), response in zip(batch, responses):
            final = restore_code(response, store)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(final, encoding="utf-8")
            log.debug("Saved: %s", out)

    log.info("SkillsBench done → %s", output_dir)

# ---------------------------------------------------------------------------
# AppWorld handler
# ---------------------------------------------------------------------------

def process_appworld(llm,
                     input_path: Path,
                     output_dir: Path,
                     lang: str,
                     task: str,
                     batch_size: int,
                     limit: int | None,
                     temperature: float,
                     max_tokens: int,
                     api_url: str | None = None,
                     api_model: str = MODEL_PATH,
                     ) -> None:
    """Transform AppWorld task instructions (appworld_tasks.jsonl: task_id + instruction).

    Output: <output_dir>/appworld_tasks_{lang}_{task}.jsonl
    Added field per record: instruction_{lang} — transformed instruction.
    """
    records = load_jsonl(input_path)
    if limit:
        records = records[:limit]

    out_path = output_dir / f"{input_path.stem}_{lang}_{task}.jsonl"

    # Resume: reload any existing output
    if out_path.exists():
        existing = {r["task_id"]: r for r in load_jsonl(out_path)}
        records = [existing.get(r["task_id"], r) for r in records]
        log.info("Resuming: %d already in output", len(existing))

    sys_prompt = get_system_prompt("appworld", task, lang)
    i_key = f"instruction_{lang}"

    pending = [(i, r) for i, r in enumerate(records) if i_key not in r]
    log.info("AppWorld %s→%s: %d records need instruction translation", task, lang, len(pending))

    for batch in tqdm(list(chunks(pending, batch_size)), desc=f"appworld/{lang}/{task}"):
        convs = [make_conversation(sys_prompt, r["instruction"]) for _, r in batch]
        responses = batch_chat(llm, convs, temperature, max_tokens, api_url, api_model)

        for j, (i, _) in enumerate(batch):
            records[i][i_key] = clean_appworld_instruction(
                records[i]["instruction"],
                responses[j],
                lang,
                task,
            )

        save_jsonl(records, out_path)

    for r in records:
        if i_key in r:
            r[i_key] = clean_appworld_instruction(r["instruction"], r[i_key], lang, task)

    save_jsonl(records, out_path)
    log.info("AppWorld done: %s", out_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multilingual transformation pipeline for IRTIQA benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dataset", choices=["drop", "snips", "skillsbench", "appworld"],
                   help="Target benchmark dataset")
    p.add_argument("task", choices=["translate", "code_switch"],
                   help="Transformation type")
    p.add_argument("lang", choices=list(LANGUAGES),
                   help="Target language code: " + ", ".join(f"{k}={v}" for k, v in LANGUAGES.items()))

    p.add_argument("--input",  default=None,
                   help="Input path (file for drop, directory for snips/skillsbench). "
                        "Defaults to the standard benchmark location.")
    p.add_argument("--output", default=None,
                   help="Output directory. Defaults to same directory as input.")
    p.add_argument("--model",  default=MODEL_PATH,
                   help="Path to the vLLM-compatible model directory.")
    p.add_argument("--tensor-parallel-size", type=int, default=1, metavar="N",
                   help="Number of GPUs for tensor parallelism (default: 1).")
    p.add_argument("--max-model-len", type=int, default=8192, metavar="N",
                   help="Maximum sequence length for vLLM (default: 8192).")
    p.add_argument("--batch-size",    type=int, default=32, metavar="N",
                   help="Inference batch size (default: 32).")
    p.add_argument("--max-tokens",    type=int, default=3000, metavar="N",
                   help="Max new tokens per response (default: 3000; covers longest DROP passages).")
    p.add_argument("--temperature",   type=float, default=0.1,
                   help="Sampling temperature; lower = more deterministic (default: 0.1).")
    p.add_argument("--limit",         type=int, default=None, metavar="N",
                   help="Process only the first N items (useful for smoke-testing).")

    # Hosted API mode (alternative to loading model in-process)
    api = p.add_argument_group("Hosted vLLM API (use instead of loading model locally)")
    api.add_argument("--api-url",   default=None,
                     help="Base URL of a running vLLM OpenAI-compatible server, "
                          "e.g. http://localhost:8000  When set, --model/--tensor-parallel-size "
                          "are ignored and the model is not loaded locally.")
    api.add_argument("--api-model", default=None,
                     help="Model name as registered on the vLLM server "
                          "(default: same as --model / the local path).")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    input_path = Path(args.input) if args.input else Path(DEFAULT_INPUT[args.dataset])
    output_dir = Path(args.output) if args.output else input_path if input_path.is_dir() else input_path.parent

    log.info("Dataset : %s", args.dataset)
    log.info("Task    : %s", args.task)
    log.info("Language: %s (%s)", args.lang, LANGUAGES[args.lang])
    log.info("Input   : %s", input_path)
    log.info("Output  : %s", output_dir)

    api_url   = args.api_url or None
    api_model = args.api_model or args.model

    llm = load_model(args.model, args.tensor_parallel_size, args.max_model_len, api_url)

    if api_url:
        log.info("Mode    : API → %s  (model: %s)", api_url, api_model)
    else:
        log.info("Mode    : local vLLM (tensor_parallel=%d)", args.tensor_parallel_size)

    kw = dict(
        llm=llm,
        lang=args.lang,
        task=args.task,
        batch_size=args.batch_size,
        limit=args.limit,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_url=api_url,
        api_model=api_model,
    )

    if args.dataset == "drop":
        process_drop(input_path=input_path, output_dir=output_dir, **kw)
    elif args.dataset == "snips":
        process_snips(input_dir=input_path, output_dir=output_dir, **kw)
    elif args.dataset == "skillsbench":
        process_skillsbench(input_dir=input_path, output_dir=output_dir, **kw)
    elif args.dataset == "appworld":
        process_appworld(input_path=input_path, output_dir=output_dir, **kw)

if __name__ == "__main__":
    main()
