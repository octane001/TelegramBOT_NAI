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
from openai import OpenAI
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

📰 हेडलाइन
4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।

📰 हेडलाइन
4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।

📰 हेडलाइन
4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।

📰 हेडलाइन
4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।

📰 हेडलाइन
4–5 छोटे, स्पष्ट और प्रभावशाली वाक्यों में समाचार का सार।




नियम:
- हर सारांश अधिकतम 50-55 शब्दों का हो।
- कुल स्क्रिप्ट 220 शब्दों से अधिक न हो।
- वाक्य छोटे और बोलने में स्वाभाविक हों।
- “इस बीच,” “वहीं,” “और अंत में,” जैसे प्राकृतिक संक्रमण शब्दों का उपयोग करें।
- दोहराव न करें।
- अतिरंजना से बचें।
- कोई markdown या ** चिन्ह न प्रयोग करें।
- केवल सादा टेक्स्ट में उत्तर दें।
- अंतिम पंक्ति पूर्ण और स्वाभाविक हो।
"""
    else:
        system_prompt = """
You are an energetic and professional English news anchor.

Your task is to create a structured news bulletin with EXACTLY 5 news stories.

STRICT FORMAT:

Start with: "Here are the five biggest stories you need to know today."
End with: "That wraps up today's top stories."

📰 Headline
5–6 short, clear, and impactful sentences summarizing the story.

📰 Headline
5–6 short, clear, and impactful sentences summarizing the story.

📰 Headline
5–6 short, clear, and impactful sentences summarizing the story.

📰 Headline
5–6 short, clear, and impactful sentences summarizing the story.

📰 Headline
5–6 short, clear, and impactful sentences summarizing the story.



RULES:
- Each summary must be a maximum of 50–55 words.
- The total script must not exceed 220 words.
- Sentences should be short and natural for spoken delivery.
- Use smooth transitions such as “Meanwhile,” “In other developments,” and “Finally,”.
- Avoid repetition.
- Avoid exaggeration.
- Do NOT use markdown or bold symbols like **.
- Output plain text only.
- Ensure the final line feels complete and natural.
"""

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nNews Content:\n{combined_text}"
            }
        ],
        temperature=0.5,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()


async def text_to_audio(text, filename, language):

    voice = "hi-IN-SwaraNeural" if language == "hindi" else "en-IN-NeerjaNeural"

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate="+10%",
)

    await communicate.save(filename)


def clean_for_output(text):
    # Remove markdown bold/italic markers
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


async def send_long_message(update, text):
    max_length = 4000
    for i in range(0, len(text), max_length):
        await update.message.reply_text(text[i:i + max_length])

# ==============================
# Subscribe for daily News feature
# ==============================

SUBSCRIBERS_FILE = "subscribers.json"

def load_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return {}
    with open(SUBSCRIBERS_FILE,"r") as f:
        return json.load(f)

def save_subscribers(data):
    with open (SUBSCRIBERS_FILE, "w") as f:
        json.dump(data,f)


# ==============================
# TELEGRAM HANDLERS
# ==============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI News Bot Ready!\n\nUse:\n/briefing <topic> <language>\n\nExample:\n/briefing AI en\n/briefing India hi"
    )
def cleanup_files(*files):
    for file in files:
        if file and os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Failed to delete {file}:", e)

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
        await update.message.reply_text("No news found for this topic.")
        return

    clean_script = clean_for_output(script)
    await send_long_message(update, clean_script)

    # Generate audio
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    voice_file = os.path.join(OUTPUT_DIR, f"voice_{timestamp}.mp3")
    final_file = os.path.join(OUTPUT_DIR, f"briefing_{timestamp}.mp3")

    await text_to_audio(clean_script, voice_file, language)
    final_audio = add_background_music(voice_file, final_file)

    if os.path.exists(final_audio):
        try:
            with open(final_audio, "rb") as audio:
                await update.message.reply_document(audio)
        except Exception as e:
            print("Audio upload failed:", e)
        finally:
        # Always delete files after attempt
            cleanup_files(voice_file, final_audio)

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
    subscribers[user_id] = {
        "topic": topic,
        "language": language
    }

    save_subscribers(subscribers)

    await update.message.reply_text(
        f"✅ Subscribed!\nYou will receive daily news on '{topic}' every morning at 9 AM."
    )

async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):

    user_id = str(update.effective_chat.id)
    subscribers = load_subscribers()

    if user_id in subscribers:
        del subscribers[user_id]
        save_subscribers(subscribers)
        await update.message.reply_text("❌ You have been unsubscribed.")
    else:
        await update.message.reply_text("You are not subscribed.")
        

async def send_daily_news(application):

    print("Scheduler triggered at:", datetime.datetime.now())

    subscribers = load_subscribers()

    for user_id, data in subscribers.items():
        try:
            topic = data["topic"]
            language = data.get("language")

            script = await generate_news_script(topic, language)
            if not script:
                continue

            clean_script = clean_for_output(script)

            # Send text
            await application.bot.send_message(
                chat_id=user_id,
                text=clean_script
            )

            # Generate Audio
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            voice_file = os.path.join(OUTPUT_DIR, f"voice_{timestamp}.mp3")
            final_file = os.path.join(OUTPUT_DIR, f"final_{timestamp}.mp3")

            await text_to_audio(clean_script, voice_file, language)

            # Add Background Music
            final_audio = add_background_music(voice_file, final_file)

            # Send Audio Properly
            with open(final_audio, "rb") as audio:
                await application.bot.send_audio(
                    chat_id=user_id,
                    audio=audio
                )

            # Cleanup
            cleanup_files(voice_file, final_audio)

        except Exception as e:
            print("Error sending to", user_id, e)
        
            
async def post_init(application):
    scheduler = AsyncIOScheduler(timezone=timezone("Asia/Kolkata"))

    scheduler.add_job(
        send_daily_news,
        "cron",
        hour=8,
        minute=59,
        args=[application]
    )

    scheduler.start()
    print("Scheduler started...")

# ==============================
# MAIN
# ==============================

def main():
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
    
    print("Bot running...")

    while True:
        try:
            app.run_polling()
        except Exception as e:
            print("Network error, retrying in 5 seconds...", e)
            import time
            time.sleep(5)

if __name__ == "__main__":
    main()