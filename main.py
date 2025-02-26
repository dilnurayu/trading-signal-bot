import time
from typing import Dict

from mysql.connector import pooling
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, PreCheckoutQuery, CallbackQuery
from telegram.ext import (
    CommandHandler, CallbackQueryHandler, Application,
    ContextTypes, MessageHandler, filters, PreCheckoutQueryHandler
)
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import mysql.connector
from datetime import datetime, timedelta
from formula import ShortTermPredictor
from localisation import LOCALIZATION, DIRECTION_TRANSLATIONS, CONFIDENCE_TRANSLATIONS



logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TOKEN = "7464820680:AAHeYmWzf88-7KDs8BiJF8liG6sQDeN31Zc"
BOT_USERNAME = "@signal_trading_ai_bot"
user_choices: Dict[int, Dict] = {}


class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.conn = mysql.connector.connect(
                    host='develosh.beget.tech',
                    user='develosh_trading',
                    password='Sher2004',
                    database='develosh_trading'
                )
                self.cursor = self.conn.cursor(dictionary=True)
                logger.info("Successfully connected to database")
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}")
            raise

    def ensure_connection(self):
        try:
            if self.conn is None or not self.conn.is_connected():
                self.connect()
        except mysql.connector.Error as err:
            logger.error(f"Database reconnection error: {err}")
            raise

    def get_user(self, user_id: int) -> dict:
        try:
            self.ensure_connection()
            self.cursor.execute(
                "SELECT * FROM users WHERE id = %s", (user_id,)
            )
            return self.cursor.fetchone()
        except mysql.connector.Error as err:
            logger.error(f"Database query error: {err}")
            raise

    def create_user(self, user_id: int, language: int = 0) -> None:
        try:
            self.ensure_connection()
            self.cursor.execute(
                """INSERT INTO users (id, available_free_request, subscribe_type, created_at, lang) 
                VALUES (%s, 3, 0, NOW(), %s)""", (user_id, language)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database insert error: {err}")
            self.conn.rollback()
            raise

    def update_user_language(self, user_id: int, language: int) -> None:
        try:
            self.ensure_connection()
            self.cursor.execute(
                "UPDATE users SET lang = %s WHERE id = %s",
                (language, user_id)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database update language error: {err}")
            self.conn.rollback()
            raise

    def decrease_available_request(self, user_id: int) -> None:
        try:
            self.ensure_connection()
            self.cursor.execute(
                "UPDATE users SET available_free_request = available_free_request - 1 WHERE id = %s",
                (user_id,)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database update error: {err}")
            self.conn.rollback()
            raise

    def add_subscription(self, user_id: int, sub_type: int, duration_days: int) -> None:
        try:
            self.ensure_connection()
            now = datetime.now()
            end_date = now + timedelta(days=duration_days)
            self.cursor.execute(
                """UPDATE users SET 
                subscribe_type = %s,
                subscribed_at = %s,
                subscribe_end = %s 
                WHERE id = %s""",
                (sub_type, now, end_date, user_id)
            )
            self.conn.commit()
        except mysql.connector.Error as err:
            logger.error(f"Database subscription error: {err}")
            self.conn.rollback()
            raise

    def check_subscription_status(self, user_id: int) -> bool:
        try:
            self.ensure_connection()
            self.cursor.execute(
                """SELECT subscribe_type, subscribe_end 
                FROM users WHERE id = %s""",
                (user_id,)
            )
            result = self.cursor.fetchone()
            if not result or not result['subscribe_type']:
                return False
            if result['subscribe_end'] is None:
                return False
            return result['subscribe_end'] > datetime.now()
        except mysql.connector.Error as err:
            logger.error(f"Database subscription check error: {err}")
            return False

    def __del__(self):
        try:
            if hasattr(self, 'cursor') and self.cursor:
                self.cursor.close()
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


db = DatabaseManager()

class PredictionManager:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def run_prediction(self, crypto: str, minutes: int) -> tuple:
        try:
            predictor = ShortTermPredictor(crypto, minutes)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.thread_pool, predictor.train_model)

            prediction_result = await loop.run_in_executor(
                self.thread_pool, predictor.predict_next_movement
            )
            return prediction_result
        except Exception as e:
            logger.error(f"Prediction error for {crypto}: {str(e)}")
            return None

prediction_manager = PredictionManager()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [
            InlineKeyboardButton("🇺🇸 English", callback_data="lang_en"),
            InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru"),
            InlineKeyboardButton("🇺🇿 O'zbek", callback_data="lang_uz")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "🌐 Please select your language / Выберите язык / Tilni tanlang:",
        reply_markup=reply_markup
    )


async def crypto(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user = db.get_user(user_id)

    if not user:
        db.create_user(user_id)
        user = db.get_user(user_id)

    if not user:
        await update.message.reply_text("❌ Error retrieving user data. Please try again later.")
        return

    lang_code_map = {0: 'en', 1: 'ru', 2: 'uz'}
    lang_code = lang_code_map.get(user.get('lang', 0), 'en')
    user_choices[user_id] = {"lang": lang_code}

    if user.get('available_free_request', 0) <= 0 and not db.check_subscription_status(user_id):
        keyboard = [[InlineKeyboardButton(LOCALIZATION[lang_code]['subscribe_button'], callback_data="subscribe")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            LOCALIZATION[lang_code]['no_free_predictions'],
            reply_markup=reply_markup
        )
    else:
        await send_crypto_options(update)


async def handle_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    lang_code = query.data.split('_')[1]

    lang_map = {'en': 0, 'ru': 1, 'uz': 2}
    lang_numeric = lang_map[lang_code]

    user = db.get_user(user_id)
    if not user:
        db.create_user(user_id, lang_numeric)
    else:
        db.update_user_language(user_id, lang_numeric)

    user_choices[user_id] = {"lang": lang_code}

    if user and user['available_free_request'] <= 0 and not db.check_subscription_status(user_id):
        keyboard = [
            [InlineKeyboardButton(LOCALIZATION[lang_code]['subscribe_button'], callback_data="subscribe")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            LOCALIZATION[lang_code]['no_free_predictions'],
            reply_markup=reply_markup
        )
    else:
        await send_crypto_options(query)


async def send_crypto_options(update: Update | CallbackQuery) -> None:
    if isinstance(update, CallbackQuery):
        user_id = update.from_user.id
        message_func = update.edit_message_text
    else:
        user_id = update.message.from_user.id
        message_func = update.message.reply_text

    lang_code = user_choices.get(user_id, {}).get('lang', 'en')

    keyboard = [
        [
            InlineKeyboardButton("₿ BTC-USD", callback_data="BTC-USD"),
            InlineKeyboardButton("Ξ ETH-USD", callback_data="ETH-USD")
        ],
        [
            InlineKeyboardButton("◎ SOL-USD", callback_data="SOL-USD"),
            InlineKeyboardButton("✧ XRP-USD", callback_data="XRP-USD")
        ],
        [
            InlineKeyboardButton("Ð DOGE-USD", callback_data="DOGE-USD"),
            InlineKeyboardButton("● DOT-USD", callback_data="DOT-USD")
        ],
        [
            InlineKeyboardButton("⚡ ADA-USD", callback_data="ADA-USD"),
            InlineKeyboardButton("∞ MATIC-USD", callback_data="MATIC-USD")
        ],
        [
            InlineKeyboardButton("⚪ LINK-USD", callback_data="LINK-USD"),
            InlineKeyboardButton("⭕ AVAX-USD", callback_data="AVAX-USD")
        ],
        [
            InlineKeyboardButton("✦ ATOM-USD", callback_data="ATOM-USD"),
            InlineKeyboardButton("◆ UNI-USD", callback_data="UNI-USD")
        ],
        [
            InlineKeyboardButton("★ NEAR-USD", callback_data="NEAR-USD"),
            InlineKeyboardButton("◈ ALGO-USD", callback_data="ALGO-USD")
        ],
        [
            InlineKeyboardButton("⬡ FTM-USD", callback_data="FTM-USD")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await message_func(
        LOCALIZATION[lang_code]['welcome'],
        reply_markup=reply_markup
    )


async def handle_subscription(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    lang_code = user_choices.get(user_id, {}).get('lang', 'en')

    prices = [LabeledPrice("Bot Subscription", 250)]
    await context.bot.send_invoice(
        chat_id=query.from_user.id,
        title="Crypto Signal Bot Subscription",
        description="30 days of full access to predictions",
        payload="subscription_30_days",
        provider_token="YOUR_PAYMENT_PROVIDER_TOKEN",
        currency="XTR",
        prices=prices
    )


async def pre_checkout_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query: PreCheckoutQuery = update.pre_checkout_query
    await query.answer(ok=True)


async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    lang_code = user_choices.get(user_id, {}).get('lang', 'en')

    db.add_subscription(user_id, 1, 30)
    await update.message.reply_text(LOCALIZATION[lang_code]['subscription_thanks'])


async def crypto_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    user = db.get_user(user_id)
    lang_code = user_choices.get(user_id, {}).get('lang', 'en')

    if not db.check_subscription_status(user_id) and user['available_free_request'] <= 0:
        await query.edit_message_text(
            LOCALIZATION[lang_code]['no_free_predictions']
        )
        return

    user_choices[user_id]["crypto"] = query.data
    keyboard = [
        [
            InlineKeyboardButton("🕐 5 Minutes", callback_data="5"),
            InlineKeyboardButton("🕑 15 Minutes", callback_data="15")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        LOCALIZATION[lang_code]['select_timeframe'].format(query.data),
        reply_markup=reply_markup
    )


async def time_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    user = db.get_user(user_id)
    lang_code = user_choices.get(user_id, {}).get('lang', 'en')

    if not db.check_subscription_status(user_id) and user['available_free_request'] <= 0:
        await query.edit_message_text(
            LOCALIZATION[lang_code]['no_free_predictions']
        )
        return

    if user_id not in user_choices:
        await query.edit_message_text(LOCALIZATION[lang_code]['session_expired'])
        return

    crypto = user_choices[user_id]["crypto"]
    minutes_ahead = int(query.data)
    user_choices[user_id]["time"] = minutes_ahead

    await query.edit_message_text(
        LOCALIZATION[lang_code]['analyzing'].format(crypto)
    )

    try:
        prediction_result = await prediction_manager.run_prediction(crypto, minutes_ahead)

        if not prediction_result:
            await query.edit_message_text(LOCALIZATION[lang_code]['prediction_error'])
            return
        prediction, probabilities, current_price, current_time, confidence_level = prediction_result

        if not db.check_subscription_status(user_id):
            db.decrease_available_request(user_id)
            user = db.get_user(user_id)

        direction = "UP" if prediction == 1 else "DOWN"
        translated_direction = DIRECTION_TRANSLATIONS[lang_code][direction]
        probability = probabilities[1] * 100 if prediction == 1 else probabilities[0] * 100
        translated_confidence = CONFIDENCE_TRANSLATIONS[lang_code][confidence_level]

        confidence_info = LOCALIZATION[lang_code]['confidence_levels'][confidence_level]
        confidence_emoji = {
            'High': '🟢',
            'Medium': '🟡',
            'Low': '🟠',
            'Very Low': '🔴'
        }[confidence_level]

        result_message = LOCALIZATION[lang_code]['prediction_result'].format(
            crypto=crypto,
            time=current_time.strftime('%Y-%m-%d %H:%M:%S'),
            price=float(current_price),
            direction=translated_direction,
            probability=probability,
            emoji=confidence_emoji,
            confidence=translated_confidence,
            minutes=minutes_ahead,
            timeframe_context=LOCALIZATION[lang_code]['timeframes'][minutes_ahead],
            recommendation=confidence_info['recommendation']
        )

        if not db.check_subscription_status(user_id):
            remaining = user['available_free_request']
            if remaining > 0:
                result_message += LOCALIZATION[lang_code]['remaining_predictions'].format(remaining)
            else:
                result_message += LOCALIZATION[lang_code]['last_prediction']

        result_message += LOCALIZATION[lang_code]['make_new_prediction']

        await query.edit_message_text(result_message)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        await query.edit_message_text(LOCALIZATION[lang_code]['prediction_error'])


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)

    if context.user_data:
        logger.error(f"User data: {context.user_data}")
    if context.chat_data:
        logger.error(f"Chat data: {context.chat_data}")

    message = "⚠️ An unexpected error occurred. Please try again later."
    try:
        if update and hasattr(update, "message") and update.message:
            await update.message.reply_text(message)
        elif update and hasattr(update, "callback_query") and update.callback_query:
            await update.callback_query.answer(message, show_alert=True)
    except Exception as e:
        logger.error(f"Error in error handler: {e}")


def main() -> None:
    logger.info("Starting bot initialization...")

    try:
        app = Application.builder().token(TOKEN).build()

        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("crypto", crypto))
        app.add_handler(CallbackQueryHandler(handle_language_selection, pattern="^lang_"))
        app.add_handler(
            CallbackQueryHandler(crypto_selection,
                                 pattern="^(BTC-USD|ETH-USD|SOL-USD|XRP-USD|DOGE-USD|DOT-USD|ADA-USD|MATIC-USD|LINK-USD|AVAX-USD|ATOM-USD|UNI-USD|NEAR-USD|ALGO-USD|FTM-USD)$"))
        app.add_handler(CallbackQueryHandler(time_selection, pattern="^(5|15)$"))
        app.add_handler(CallbackQueryHandler(handle_subscription, pattern="^subscribe$"))
        app.add_handler(PreCheckoutQueryHandler(pre_checkout_query_handler))
        app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))

        app.add_error_handler(error_handler)

        logger.info("🚀 Bot is starting...")
        logger.info(f"Bot username: {BOT_USERNAME}")

        app.run_polling()

    except Exception as e:
        logger.critical(f"Failed to start bot: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()