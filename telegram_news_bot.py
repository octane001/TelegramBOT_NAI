import re
import os
import json
import datetime
import asyncio
import feedparser
import edge_tts
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from openai import OpenAI, APIStatusError, APIConnectionError
from dotenv import load_dotenv
from pydub import AudioSegment
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone

load_dotenv()

# ==============================
# CONFIG
# ==============================

TOKEN = os.getenv("TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not TOKEN:
    raise ValueError("❌ TOKEN not found in .env file!")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found in .env file!")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Telegram News Bot"
    }
)

MAX_ARTICLES = 5
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "inclusionai/ling-2.6-flash:free"

# Lock to prevent simultaneous writes to subscribers.json
_subscribers_lock = asyncio.Lock()

# ==============================
# UTIL FUNCTIONS
# ==============================

def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def fetch_news(topic):
    encoded_topic = quote_plus(topic)
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    return feed.entries[:MAX_ARTICLES]


async def generate_news_script(topic, language):
    entries = fetch_news(topic)

    if not entries:
        print(f"[WARN] No RSS entries found for topic: {topic}")
        return None

    article_blocks = []
    for entry in entries:
        title = clean_html(entry.title)
        summary = clean_html(entry.get("summary", ""))
        article_blocks.append(f"{title}\n{summary}")

    combined_text = "\n\n".join(article_blocks)

    if language == "hindi":
        system_prompt = """
आप एक ऊर्जावान और पेशेवर हिंदी न्यूज़ एंकर हैं।

आपका कार्य बिल्कुल 5 समाचारों का संरचित बुलेटिन बनाना है।

सख्त फ़ॉर्मेट:


शुरुआत करें: "आज की 5 बड़ी खबरें जो आपको जाननी चाहिए।"
समापन करें: "फिलहाल के लिए बस इतना ही।"

4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।


4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।


4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।


4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।


4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।




नियम:
- हर सारांश अधिकतम 50-55 शब्दों का हो।
- कुल स्क्रिप्ट 220 शब्दों से अधिक न हो।
- वाक्य छोटे और बोलने में स्वाभाविक हों।
- "इस बीच," "वहीं," "और अंत में," जैसे प्राकृतिक संक्रमण शब्दों का उपयोग करें।
- दोहराव न करें।
- अतिरंजना से बचें।
- कोई markdown या ** चिन्ह न प्रयोग करें।
- केवल सादा टेक्स्ट में उत्तर दें।
- अंतिम पंक्ति पूर्ण और स्वाभाविक हो।
"""
    else:
        system_prompt = """
You are a world-class professional TV news anchor and scriptwriter.

Your task is to create a highly engaging, polished, natural-sounding audio news bulletin using the provided news content.

Generate EXACTLY 5 important news stories.

GOAL:
The script must sound like a premium modern news broadcast that people enjoy listening to on Telegram, YouTube, Spotify, or radio.

STRICT OUTPUT FORMAT:

Start exactly with:
Here are the five biggest stories you need to know today.

Then produce EXACTLY 5 stories in this format:

📰 Short Powerful Headline

2 to 4 short spoken sentences explaining:
- what happened
- why it matters
- latest update or impact

Use natural transitions between stories such as:
Meanwhile,
In other news,
Turning to business,
Elsewhere,
And finally,

End exactly with:
That wraps up today's top stories.

STRICT RULES:

1. Total output between 180 and 260 words.
2. Keep sentences short, sharp, and easy to hear.
3. Write for the ear, not for reading.
4. Sound confident, premium, modern, and human.
5. Avoid robotic wording.
6. Avoid repeating names too many times.
7. Avoid exaggerated clickbait language.
8. Make every story feel important.
9. Use active voice.
10. No markdown except 📰
11. No bullet points.
12. No extra commentary.
13. No quotes unless essential.
14. If numbers are complex, simplify them naturally.
15. Each story should feel fresh and fast-paced.

STYLE REFERENCE:

Think BBC + Bloomberg + modern YouTube news energy.

VERY IMPORTANT:

This script will be converted to speech, so rhythm, clarity, and listener retention matter more than fancy writing.
"""

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\n\nNews Content:\n{combined_text}"
                }
            ],
            temperature=0.55,
            max_tokens=700
        )
        content = response.choices[0].message.content
        if not content:
            print(f"[WARN] Model returned empty content. Full response: {response}")
            return None
        return content.strip()

    except APIStatusError as e:
        print(f"[API ERROR] Status {e.status_code}: {e.message}")
        if e.status_code == 401:
            print("➡ Your OPENROUTER_API_KEY is invalid or missing. Check your .env file.")
        elif e.status_code == 404:
            print(f"➡ Model '{MODEL}' not found on OpenRouter. Try a different model.")
        elif e.status_code == 429:
            print("➡ Rate limit hit. Wait a moment and try again.")
        return None

    except APIConnectionError as e:
        print(f"[CONNECTION ERROR] Could not reach OpenRouter: {e}")
        return None

    except Exception as e:
        print(f"[UNEXPECTED ERROR] {type(e).__name__}: {e}")
        return None


async def text_to_audio(text, filename, language):
    voice = "hi-IN-SwaraNeural" if language == "hindi" else "en-IN-NeerjaNeural"
    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate="+10%",
    )
    await communicate.save(filename)


def clean_for_output(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = text.replace("---", "")
    return text.strip()


def add_background_music(voice_path, output_path):
    if not os.path.exists("background.mp3"):
        return voice_path

    voice = AudioSegment.from_file(voice_path)
    music = AudioSegment.from_file("background.mp3")
    music = music - 30

    if len(music) < len(voice):
        loops = len(voice) // len(music) + 1
        music = music * loops

    music = music[:len(voice)]
    final = voice.overlay(music).fade_out(2000)
    final.export(output_path, format="mp3")
    return output_path


def cleanup_files(*files):
    for file in files:
        if file and os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete {file}:", e)


async def send_long_message(update, text):
    max_length = 4000
    for i in range(0, len(text), max_length):
        await update.message.reply_text(text[i:i + max_length])


# ==============================
# SUBSCRIBERS
# ==============================

SUBSCRIBERS_FILE = "subscribers.json"


def load_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return {}
    with open(SUBSCRIBERS_FILE, "r") as f:
        return json.load(f)


async def save_subscribers(data):
    # Prevents two users writing at the same time and corrupting the file
    async with _subscribers_lock:
        with open(SUBSCRIBERS_FILE, "w") as f:
            json.dump(data, f, indent=2)


# ==============================
# TELEGRAM HANDLERS
# ==============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI News Bot Ready!\n\nUse:\n/briefing <topic> <language>\n\nExample:\n/briefing AI en\n/briefing India hi"
    )


async def briefing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Usage:\n/briefing <topic> <language>\n\nExample:\n/briefing AI en\n/briefing India hi"
        )
        return

    if context.args[-1].lower() in ["hi", "hindi"]:
        language = "hindi"
        topic = " ".join(context.args[:-1])
    elif context.args[-1].lower() in ["en", "english"]:
        language = "english"
        topic = " ".join(context.args[:-1])
    else:
        language = "english"
        topic = " ".join(context.args)

    if not topic:
        await update.message.reply_text("Please provide a valid topic.")
        return

    await update.message.reply_text(
        f"Generating 5 news about '{topic}' in {language}... ⏳"
    )

    script = await generate_news_script(topic, language)

    if not script:
        await update.message.reply_text(
            "❌ Failed to generate news. Check the console for the exact error "
            "(likely: invalid API key, wrong model, or rate limit)."
        )
        return

    clean_script = clean_for_output(script)
    await send_long_message(update, clean_script)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    chat_id = str(update.effective_chat.id)
    voice_file = os.path.join(OUTPUT_DIR, f"voice_{chat_id}_{timestamp}.mp3")
    final_file = os.path.join(OUTPUT_DIR, f"briefing_{chat_id}_{timestamp}.mp3")

    try:
        await text_to_audio(clean_script, voice_file, language)
        final_audio = add_background_music(voice_file, final_file)

        if os.path.exists(final_audio):
            with open(final_audio, "rb") as audio:
                await update.message.reply_document(audio)
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")
        await update.message.reply_text("⚠️ Audio generation failed, but your text briefing is above.")
    finally:
        cleanup_files(voice_file, final_file)


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)

    if not context.args:
        await update.message.reply_text(
            "Usage:\n/subscribe <topic> <language>\nExample:\n/subscribe AI en"
        )
        return

    if context.args[-1].lower() in ["hi", "hindi"]:
        language = "hindi"
        topic = " ".join(context.args[:-1])
    else:
        language = "english"
        topic = " ".join(context.args)

    subscribers = load_subscribers()
    subscribers[user_id] = {"topic": topic, "language": language}
    await save_subscribers(subscribers)

    await update.message.reply_text(
        f"✅ Subscribed!\nYou will receive daily news on '{topic}' every morning at 9 AM IST."
    )


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    subscribers = load_subscribers()

    if user_id in subscribers:
        del subscribers[user_id]
        await save_subscribers(subscribers)
        await update.message.reply_text("❌ You have been unsubscribed.")
    else:
        await update.message.reply_text("You are not currently subscribed.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print(f"[TELEGRAM ERROR] {context.error}")
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(
            "⚠️ Something went wrong. Please try again in a moment."
        )


# ==============================
# DAILY SCHEDULER
# ==============================

async def send_news_to_user(application, user_id, data):
    """Send daily news to a single user — runs in parallel for all subscribers."""
    try:
        topic = data["topic"]
        language = data.get("language", "english")

        script = await generate_news_script(topic, language)
        if not script:
            return

        clean_script = clean_for_output(script)
        await application.bot.send_message(chat_id=user_id, text=clean_script)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        voice_file = os.path.join(OUTPUT_DIR, f"voice_{user_id}_{timestamp}.mp3")
        final_file = os.path.join(OUTPUT_DIR, f"final_{user_id}_{timestamp}.mp3")

        await text_to_audio(clean_script, voice_file, language)
        final_audio = add_background_music(voice_file, final_file)

        with open(final_audio, "rb") as audio:
            await application.bot.send_audio(chat_id=user_id, audio=audio)

        cleanup_files(voice_file, final_audio)

    except Exception as e:
        print(f"[DAILY ERROR] Error sending to {user_id}: {e}")


async def send_daily_news(application):
    print("Scheduler triggered at:", datetime.datetime.now())
    subscribers = load_subscribers()

    if not subscribers:
        print("No subscribers found.")
        return

    # Send to ALL users at the same time instead of one by one
    tasks = [
        send_news_to_user(application, user_id, data)
        for user_id, data in subscribers.items()
    ]
    await asyncio.gather(*tasks)
    print(f"✅ Daily news sent to {len(tasks)} subscribers.")


async def post_init(application):
    scheduler = AsyncIOScheduler(timezone=timezone("Asia/Kolkata"))
    scheduler.add_job(
        send_daily_news,
        "cron",
        hour=9,
        minute=0,
        args=[application]
    )
    scheduler.start()
    print("Scheduler started — daily news at 9:00 AM IST")


# ==============================
# MAIN for local  testing
# ==============================

# def main():
#     app = (
#         ApplicationBuilder()
#         .token(TOKEN)
#         .post_init(post_init)
#         .read_timeout(60)
#         .write_timeout(60)
#         .connect_timeout(60)
#         .pool_timeout(60)
#         .build()
#     )

#     app.add_handler(CommandHandler("start", start))
#     app.add_handler(CommandHandler("briefing", briefing))
#     app.add_handler(CommandHandler("subscribe", subscribe))
#     app.add_handler(CommandHandler("unsubscribe", unsubscribe))
#     app.add_error_handler(error_handler)

#     print("Bot running in polling mode...")
#     app.run_polling()

# ==============================
# MAIN for production render deployment
# ==============================

def main():
    port = int(os.environ.get("PORT", 10000))
    webhook_url = os.environ.get("RENDER_EXTERNAL_URL")

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .read_timeout(60)
        .write_timeout(60)
        .connect_timeout(60)
        .pool_timeout(60)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("briefing", briefing))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_error_handler(error_handler)

    print("Bot running in webhook mode...")

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        webhook_url=f"{webhook_url}/{TOKEN}",
        url_path=TOKEN,
    )
if __name__ == "__main__":
    main()